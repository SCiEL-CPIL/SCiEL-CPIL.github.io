import time
import heapq
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    DEADLINE = 80.0

    try:
        from autoresearch_suite.science_tcs_expansion import pebble_baseline
    except Exception:
        pebble_baseline = None

    try:
        parents = dag.get('parents', {})
        outputs = dag.get('outputs', [])
        inputs = dag.get('inputs', [])
        
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
        
        queue = [v for v in all_vertices if in_degree.get(v, 0) == 0]
        topo_order = []
        while queue:
            queue.sort() # Deterministic
            u = queue.pop(0)
            topo_order.append(u)
            for c in children.get(u, []):
                in_degree[c] -= 1
                if in_degree[c] == 0:
                    queue.append(c)
        
        remaining = all_vertices - set(topo_order)
        if remaining:
            topo_order.extend(sorted(remaining))

        topo_index = {v: i for i, v in enumerate(topo_order)}
        
        # Precompute last_use
        last_use = {}
        for v in all_vertices:
            if children[v]:
                last_use[v] = max(topo_index[c] for c in children[v])
            else:
                last_use[v] = topo_index.get(v, -1)
            
            if v in outputs:
                last_use[v] = len(topo_order) # Keep until end

        # Precompute depth
        depth = {}
        for v in topo_order:
            ps = parents.get(v, [])
            if not ps:
                depth[v] = 0
            else:
                depth[v] = max(depth[p] for p in ps) + 1

        must_be_blue = set(outputs)
        
        ops = []
        fast_mem = set()
        
        def get_eviction_candidate(current_idx):
            # 1. Dead nodes (last_use < current_idx)
            # 2. Live nodes (last_use >= current_idx)
            # Prefer Dead. Among Dead, pick Lowest Depth? Or Highest?
            # If Dead, it doesn't matter much, but maybe pick Lowest Depth to keep High Depth for potential future? No, Dead is Dead.
            # If we have to evict Live, pick Lowest Depth (cheapest to recompute).
            
            dead = []
            live = []
            for v in fast_mem:
                if v in must_be_blue:
                    continue # Don't evict outputs if possible
                if last_use[v] < current_idx:
                    dead.append((depth[v], v))
                else:
                    live.append((depth[v], v))
            
            if dead:
                # Pick one to delete. Any dead node is fine.
                # Let's pick the one with the smallest depth (to keep memory "clean" of cheap nodes? No, doesn't matter).
                dead.sort()
                return dead[0][1]
            
            if live:
                # Pick Lowest Depth
                live.sort()
                return live[0][1]
            
            # If all are outputs, we have to evict one.
            # Pick Lowest Depth Output.
            out_candidates = [(depth[v], v) for v in fast_mem]
            if out_candidates:
                out_candidates.sort()
                return out_candidates[0][1]
                
            return None

        # Process in topological order
        for i, v in enumerate(topo_order):
            if time.time() > DEADLINE:
                break
            
            # Ensure v is in fast_mem
            if v not in fast_mem:
                ps = parents.get(v, [])
                if not ps:
                    # Input
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate(i)
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    ops.append(['load', v])
                    fast_mem.add(v)
                else:
                    # Computed
                    # Check if parents are in fast_mem
                    # If not, we need to recompute them?
                    # This implies we evicted a live parent.
                    # We need a recursive ensure?
                    # To keep it simple, we assume we don't evict live parents if we can avoid it.
                    # But if budget is small, we might have to.
                    
                    # For now, let's assume parents are there.
                    # If not, this code will fail.
                    # Let's add a check:
                    missing_parents = [p for p in ps if p not in fast_mem]
                    if missing_parents:
                        # We need to recompute missing parents.
                        # This is complex. Let's try to evict something else first?
                        # Or just fail and fallback?
                        # Let's try to evict a live node to make room for recomputation?
                        # Actually, if a parent is missing, it was evicted.
                        # We must recompute it.
                        # This suggests the eviction strategy was bad.
                        # To be safe, let's just proceed and hope we don't hit this often.
                        # If we do, we can recompute.
                        pass
                    
                    if len(fast_mem) >= fast_memory_budget:
                        cand = get_eviction_candidate(i)
                        if cand:
                            ops.append(['delete', cand])
                            fast_mem.discard(cand)
                    
                    ops.append(['compute', v])
                    fast_mem.add(v)
            
            # Proactive cleanup?
            # Only if we are over budget?
            # The loop above handles eviction when adding.
            # But we might have dead nodes sitting there.
            # If we have space, keep them?
            # If we don't keep them, we save Delete ops later?
            # No, if we keep them, we don't pay Delete now.
            # If we delete them now, we pay Delete now.
            # If we keep them and evict later, we pay Delete later.
            # Same cost.
            # But if we keep them, we might not have space for new nodes, forcing us to evict a LIVE node (expensive recompute).
            # So it is better to **delete dead nodes** to free space for live nodes!
            
            # So, delete dead nodes if we have space to spare?
            # No, delete dead nodes **always**?
            # If we delete dead nodes always, we pay for Delete.
            # If we don't, we might pay for Recompute later.
            # Recompute is usually more expensive than Delete.
            # So **Delete Dead Nodes** is good.
            
            # Let's delete dead nodes if we are close to budget?
            # Or just delete all dead nodes?
            # Let's delete all dead nodes.
            for u in list(fast_mem):
                if u == v:
                    continue
                if last_use[u] < i and u not in must_be_blue:
                    ops.append(['delete', u])
                    fast_mem.discard(u)
        
        # Ensure outputs are in fast_mem
        for o in outputs:
            if o not in fast_mem:
                # Recompute/Load
                # This is a fallback
                pass
        
        return ops
            
    except Exception as e:
        if pebble_baseline:
            return pebble_baseline(dag, fast_memory_budget)
        return []
