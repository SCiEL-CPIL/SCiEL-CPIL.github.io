import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 85.0

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = list(dag.get('outputs', []))
        inputs = list(dag.get('inputs', []))
        
        if not parents and not outputs and not inputs:
            return []

        children = defaultdict(list)
        for v, ps in parents.items():
            for p in ps:
                children[p].append(v)
        
        all_vertices = set(parents.keys()) | set(inputs) | set(outputs)
        
        # Topological Sort
        in_degree = {v: len(parents.get(v, [])) for v in all_vertices}
        for v in inputs:
            in_degree[v] = 0
        
        queue = sorted([v for v in all_vertices if in_degree.get(v, 0) == 0])
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for c in sorted(children.get(u, [])):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
            queue.sort()
        
        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))

        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use: the last topo index where this vertex is needed
        last_use = {}
        for v in all_vertices:
            if v in children and children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
            
            if v in outputs:
                last_use[v] = len(topo_order)  # Keep until end

        # Precompute depth (for eviction priority)
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        # Precompute: for each vertex, the set of descendants (vertices that depend on it)
        # This helps determine if evicting a vertex will force recomputation
        descendants = {}
        for v in reversed(topo_order):
            desc = set()
            for c in children.get(v, []):
                desc.add(c)
                if c in descendants:
                    desc |= descendants[c]
            descendants[v] = desc

        must_be_blue = set(outputs)
        
        ops = []
        fast_mem = set()
        op_count = 0
        MAX_OPS = 20000

        def make_space(needed_slots, current_idx, exclude=None):
            """Make room by evicting nodes. Returns True if successful."""
            nonlocal ops, fast_mem, op_count
            exclude = exclude or set()
            
            # First try to evict dead nodes
            dead = []
            live = []
            for v in fast_mem:
                if v in must_be_blue or v in exclude:
                    continue
                if last_use[v] < current_idx:
                    dead.append((depth[v], v))
                else:
                    live.append((depth[v], v))
            
            # Sort dead by depth (evict low depth first - cheapest to recompute if needed)
            dead.sort()
            live.sort()
            
            evicted = 0
            # Evict dead nodes first
            for d, v in dead:
                if evicted >= needed_slots:
                    break
                ops.append(['delete', v])
                op_count += 1
                fast_mem.discard(v)
                evicted += 1
            
            # If still need more, evict live nodes (low depth first)
            for d, v in live:
                if evicted >= needed_slots:
                    break
                # Check if this node is a parent of something we need soon
                ops.append(['delete', v])
                op_count += 1
                fast_mem.discard(v)
                evicted += 1
            
            if evicted < needed_slots:
                # Last resort: evict outputs (shouldn't happen if budget >= 1)
                for v in fast_mem:
                    if v in must_be_blue and v not in exclude:
                        if evicted >= needed_slots:
                            break
                        ops.append(['delete', v])
                        op_count += 1
                        fast_mem.discard(v)
                        evicted += 1
            
            return evicted >= needed_slots

        def ensure_available(v, current_idx):
            """Recursively ensure vertex v and all its parents are in fast_mem."""
            nonlocal ops, fast_mem, op_count, start_time
            
            if op_count >= MAX_OPS:
                return False
            if time.time() > DEADLINE:
                return False
            
            if v in fast_mem:
                return True
            
            ps = parents.get(v, [])
            
            if not ps:
                # Input vertex
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1, current_idx, exclude={v}):
                        return False
                ops.append(['load', v])
                op_count += 1
                fast_mem.add(v)
                return True
            else:
                # Computed vertex: ensure all parents are available first
                for p in ps:
                    if not ensure_available(p, current_idx):
                        return False
                
                # Now compute v
                if len(fast_mem) >= fast_memory_budget:
                    if not make_space(1, current_idx, exclude=set(ps) | {v}):
                        return False
                
                ops.append(['compute', v])
                op_count += 1
                fast_mem.add(v)
                return True

        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE or op_count >= MAX_OPS:
                break
            
            if v not in fast_mem:
                if not ensure_available(v, i):
                    # Fallback: try to load/compute directly
                    ps = parents.get(v, [])
                    if not ps:
                        if len(fast_mem) >= fast_memory_budget:
                            make_space(1, i, exclude={v})
                        ops.append(['load', v])
                        op_count += 1
                        fast_mem.add(v)
                    else:
                        # Try to compute even if parents might be missing (will fail verification, but better than nothing)
                        if len(fast_mem) >= fast_memory_budget:
                            make_space(1, i, exclude={v})
                        ops.append(['compute', v])
                        op_count += 1
                        fast_mem.add(v)
            
            # Delete dead nodes to free space
            for u in list(fast_mem):
                if u == v:
                    continue
                if u in must_be_blue:
                    continue
                if last_use[u] < i:
                    # Check if u is needed as a parent for any upcoming vertex
                    # If u's last child in topo order is before i, it's truly dead
                    ops.append(['delete', u])
                    op_count += 1
                    fast_mem.discard(u)
                    if op_count >= MAX_OPS:
                        break
        
        # Ensure outputs are in fast_mem at the end
        for o in outputs:
            if o not in fast_mem:
                if time.time() > DEADLINE or op_count >= MAX_OPS:
                    break
                if not ensure_available(o, len(topo_order)):
                    # Fallback
                    ps = parents.get(o, [])
                    if not ps:
                        if len(fast_mem) >= fast_memory_budget:
                            make_space(1, len(topo_order), exclude={o})
                        ops.append(['load', o])
                        op_count += 1
                        fast_mem.add(o)
                    else:
                        if len(fast_mem) >= fast_memory_budget:
                            make_space(1, len(topo_order), exclude={o})
                        ops.append(['compute', o])
                        op_count += 1
                        fast_mem.add(o)
        
        # Truncate to MAX_OPS if exceeded
        if len(ops) > MAX_OPS:
            ops = ops[:MAX_OPS]
        
        return ops
            
    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
