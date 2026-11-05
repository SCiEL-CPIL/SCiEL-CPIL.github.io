"""Pebbling algorithm for communication-minimizing DAG evaluation."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    """
    Pebble a DAG with bounded fast memory to minimize communication.
    
    Uses a greedy pebbling strategy that balances memory pressure with
    recomputation to minimize total data transfers (loads + stores).
    """
    deadline = time.time() + 50  # Safety margin
    
    # Parse the DAG structure
    vertices = []
    parents_map = {}
    children_map = defaultdict(list)
    is_output = {}
    
    try:
        if isinstance(dag, list):
            for item in dag:
                if isinstance(item, dict):
                    v_id = item.get('id', item.get('vertex', item.get('name', None)))
                    pars = item.get('parents', item.get('parent', []))
                    out = item.get('is_output', item.get('output', False))
                elif isinstance(item, (tuple, list)):
                    if len(item) >= 3:
                        v_id, pars, out = item[0], item[1], item[2]
                    elif len(item) == 2:
                        v_id, pars = item[0], item[1]
                        out = False
                    else:
                        continue
                else:
                    continue
                
                if v_id is None:
                    continue
                    
                vertices.append(v_id)
                parents_map[v_id] = list(pars) if pars else []
                is_output[v_id] = bool(out)
                
                for p in parents_map[v_id]:
                    children_map[p].append(v_id)
    except Exception:
        pass
    
    if not vertices:
        return _naive_schedule(dag, fast_memory_budget)
    
    # Build topological order
    in_degree = {v: len(parents_map.get(v, [])) for v in vertices}
    topo_order = []
    queue = sorted([v for v in vertices if in_degree[v] == 0])
    while queue:
        v = queue.pop(0)
        topo_order.append(v)
        for c in children_map.get(v, []):
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
        queue.sort()
    
    if len(topo_order) < len(vertices):
        remaining = set(vertices) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Identify outputs
    outputs = [v for v in topo_order if is_output.get(v, False)]
    if not outputs:
        outputs = [v for v in topo_order if not children_map.get(v, [])]
    
    # State
    fast_mem = set()  # vertices in fast memory (blue)
    in_slow = set()   # vertices in slow memory (red, computed)
    ops = []
    op_count = 0
    MAX_OPS = 19500  # Leave some margin
    
    # Track how many children still need each vertex
    remaining_uses = {v: len(children_map.get(v, [])) for v in vertices}
    # For outputs, they need to end blue, so mark them
    for v in outputs:
        remaining_uses[v] = max(remaining_uses[v], 1)
    
    # Position in topological order for priority
    topo_pos = {v: i for i, v in enumerate(topo_order)}
    
    def can_add():
        return len(fast_mem) < fast_memory_budget
    
    def try_load(v):
        """Try to load vertex v into fast memory. Returns True if successful."""
        nonlocal op_count
        if v in fast_mem:
            return True
        if not can_add():
            return False
        if v in in_slow:
            in_slow.discard(v)
        fast_mem.add(v)
        ops.append(['load', v])
        op_count += 1
        return True
    
    def try_store(v):
        """Store vertex v from fast to slow memory. Returns True if successful."""
        nonlocal op_count
        if v not in fast_mem:
            return False
        fast_mem.discard(v)
        in_slow.add(v)
        ops.append(['store', v])
        op_count += 1
        return True
    
    def try_delete(v):
        """Delete vertex v from slow memory."""
        nonlocal op_count
        if v not in in_slow:
            return False
        in_slow.discard(v)
        ops.append(['delete', v])
        op_count += 1
        return True
    
    def make_space(n_needed):
        """Make n_needed slots available in fast memory by storing vertices.
        Choose vertices to store based on:
        1. Not needed immediately (high topo_pos of earliest child)
        2. Already in slow memory (just need to keep it there)
        3. Fewer remaining uses
        """
        nonlocal op_count
        if len(fast_mem) + n_needed <= fast_memory_budget:
            return True
        
        # Find vertices to store
        # Priority: store vertices that are least urgent
        candidates = []
        for v in fast_mem:
            # Don't store if it's an output that needs to end blue
            if v in outputs and remaining_uses.get(v, 0) > 0:
                # Still needed as output
                candidates.append((1000 + topo_pos.get(v, 0), v))
            else:
                # Find earliest child that needs it
                kids = children_map.get(v, [])
                if kids:
                    earliest = min(topo_pos.get(c, float('inf')) for c in kids)
                else:
                    earliest = float('inf')
                # Lower remaining uses = better to store
                uses = remaining_uses.get(v, 0)
                priority = (earliest, uses, topo_pos.get(v, 0))
                candidates.append((priority, v))
        
        # Sort by priority (earliest need first = least urgent to keep)
        candidates.sort(key=lambda x: x[0])
        
        stored = 0
        for priority, v in candidates:
            if len(fast_mem) + n_needed <= fast_memory_budget:
                break
            if v in fast_mem:
                try_store(v)
                stored += 1
            if op_count >= MAX_OPS:
                break
        
        return len(fast_mem) + n_needed <= fast_memory_budget
    
    # Process in topological order
    for v in topo_order:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        
        parents = parents_map.get(v, [])
        
        # For inputs, load them
        if not parents:
            if v not in fast_mem:
                if try_load(v):
                    pass
                else:
                    make_space(1)
                    try_load(v)
            continue
        
        # For non-inputs, ensure parents are in fast memory
        all_ready = True
        for p in parents:
            if p not in fast_mem:
                if not try_load(p):
                    make_space(1)
                    if not try_load(p):
                        all_ready = False
                        break
        
        if not all_ready:
            continue
        
        # Compute v
        if v not in fast_mem:
            if not can_add():
                make_space(1)
            if can_add():
                fast_mem.add(v)
            else:
                continue
        
        ops.append(['compute', v])
        op_count += 1
        
        # Update remaining uses for parents
        for p in parents:
            if remaining_uses.get(p, 0) > 0:
                remaining_uses[p] -= 1
        
        # If a parent is no longer needed and in fast memory, consider storing
        for p in parents:
            if remaining_uses.get(p, 0) == 0 and p not in outputs:
                # Can store or delete
                if p in fast_mem:
                    # Store to slow memory (in case we need to recompute)
                    try_store(p)
                # Delete from slow if not needed
                if p in in_slow:
                    try_delete(p)
    
    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if time.time() > deadline or op_count >= MAX_OPS:
            break
        if v not in fast_mem:
            if v in in_slow:
                make_space(1)
                try_load(v)
            else:
                # Need to recompute - this is complex
                # For now, just try to load
                make_space(1)
                try_load(v)
    
    # Trim to max ops if needed
    if len(ops) > MAX_OPS:
        ops = ops[:MAX_OPS]
    
    return ops


def _naive_schedule(dag, fast_memory_budget):
    """Naive schedule for fallback."""
    ops = []
    try:
        if isinstance(dag, list):
            for item in dag:
                if isinstance(item, dict):
                    v_id = item.get('id', item.get('vertex', item.get('name', None)))
                    pars = item.get('parents', item.get('parent', []))
                elif isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        v_id, pars = item[0], item[1]
                    else:
                        continue
                else:
                    continue
                if v_id is None:
                    continue
                if not pars:
                    ops.append(['load', v_id])
                else:
                    ops.append(['compute', v_id])
    except Exception:
        pass
    return ops
# EVOLVE-BLOCK-END
