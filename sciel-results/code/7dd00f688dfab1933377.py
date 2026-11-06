"""Rewritten program for Communication-minimizing DAG pebbling."""
# EVOLVE-BLOCK-START
import time
import sys

def pebble(dag, fast_memory_budget):
    """
    Minimize communication (load/store) operations for DAG pebbling.
    Returns: list of [operation, vertex] where operation in {load, store, compute, delete}
    """
    deadline = time.time() + 50  # Safety margin before 60s limit

    # Parse DAG structure
    if isinstance(dag, dict):
        vertices = dag.get('vertices', dag.get('nodes', dag.get('V', None)))
        parents = dag.get('parents', dag.get('edges', dag.get('pred', None)))
        inputs = dag.get('inputs', dag.get('sources', dag.get('input_nodes', None)))
        outputs = dag.get('outputs', dag.get('sinks', dag.get('output_nodes', None)))
    else:
        vertices = None
        parents = None
        inputs = None
        outputs = None

    if vertices is None or parents is None:
        try:
            from autoresearch_suite.science_tcs_expansion import pebble_baseline
            return pebble_baseline(dag, fast_memory_budget)
        except Exception:
            return []

    # Normalize to sets/lists
    if isinstance(vertices, (list, tuple, range)):
        vertex_list = list(vertices)
    elif isinstance(vertices, int):
        vertex_list = list(range(vertices))
    else:
        vertex_list = list(vertices)

    num_vertices = len(vertex_list)

    # Normalize parents to dict
    if isinstance(parents, dict):
        parent_dict = dict(parents)
    elif isinstance(parents, (list, tuple)):
        parent_dict = {i: list(p) for i, p in enumerate(parents)}
    else:
        parent_dict = {}

    for v in vertex_list:
        if v not in parent_dict:
            parent_dict[v] = []

    # Normalize inputs and outputs
    if inputs is None:
        inputs = set(v for v in vertex_list if not parent_dict.get(v, []))
    elif isinstance(inputs, (set, list, tuple)):
        inputs = set(inputs)
    else:
        inputs = set(inputs)

    if outputs is None:
        children = {v: [] for v in vertex_list}
        for v in vertex_list:
            for p in parent_dict.get(v, []):
                children[p].append(v)
        outputs = set(v for v in vertex_list if not children[v])
    elif isinstance(outputs, (set, list, tuple)):
        outputs = set(outputs)
    else:
        outputs = set(outputs)

    # Build children map
    children = {v: [] for v in vertex_list}
    for v in vertex_list:
        for p in parent_dict.get(v, []):
            children[p].append(v)

    # Determine needed vertices (ancestors of outputs)
    needed = set(outputs)
    for v in reversed(vertex_list):
        if v in needed:
            for p in parent_dict.get(v, []):
                needed.add(p)

    needed_vertices = [v for v in vertex_list if v in needed]
    needed_set = set(needed_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}

    # Compute last use index for each vertex
    last_child_idx = {}
    for v in needed_vertices:
        last_child_idx[v] = -1
        for child in children.get(v, []):
            if child in needed_set:
                last_child_idx[v] = max(last_child_idx[v], vertex_to_idx.get(child, -1))

    # State
    in_fast = set()
    in_slow = set(inputs)  # Inputs start in slow memory
    computed = set()
    operations = []

    def load(v):
        operations.append(['load', v])
        in_fast.add(v)
        # If loaded from slow, it's still in slow.
        # If loaded from nowhere (recomputation scenario? No, load implies it exists in slow).
        # We assume load only happens if v is in_slow or was previously stored.
        
    def store(v):
        operations.append(['store', v])
        in_fast.discard(v)
        in_slow.add(v)
        
    def compute(v):
        operations.append(['compute', v])
        computed.add(v)
        in_fast.add(v)
        # Newly computed values are NOT in slow memory yet.
        
    def delete(v):
        operations.append(['delete', v])
        in_fast.discard(v)
        # Do not add to in_slow. If it was in slow, it remains there.
        # If it was not in slow, it is lost.

    # Process vertices in topological order
    for v in needed_vertices:
        if time.time() > deadline:
            break

        parents = [p for p in parent_dict.get(v, []) if p in needed_set]
        v_idx = vertex_to_idx.get(v, -1)

        # Ensure parents are in fast memory
        for p in parents:
            if p in in_fast:
                continue
            
            # Need to load parent p
            if len(in_fast) >= fast_memory_budget:
                # Evict: Find best candidate to remove
                # Candidates: all in_fast except p (if p was in fast, we wouldn't be here)
                # and except other parents of v that are already in fast? No, we can evict them if needed, but we prefer to keep them.
                # Actually, we should not evict parents that we haven't loaded yet? 
                # If we evict a parent q of v that is in fast, we will have to reload it.
                # So we should prefer to evict non-parents.
                
                evicted = False
                # 1. Try to evict a vertex that is no longer needed (last_use < v_idx)
                #    AND is in slow memory (so we can delete it cheaply)
                #    OR is not in slow (so we must store it)
                
                best_evict = None
                best_score = -1
                best_is_slow = False
                
                for u in list(in_fast):
                    if u == p: continue # We are loading p, so we don't evict p (it's not in fast anyway)
                    # Prefer not to evict parents of v that we still need to load? 
                    # If u is a parent of v, and we evict it, we have to reload it.
                    # So we give a penalty to evicting parents of v.
                    is_parent_of_v = (u in parents)
                    
                    u_last = last_child_idx.get(u, -1)
                    # We want to evict the one with the FURTHEST last use.
                    # If last use is in the past (u_last < v_idx), it's a great candidate.
                    
                    # Score: Higher is better to evict.
                    # 1. If u_last < v_idx, it's useless. Score = 1000 + u_last
                    # 2. If u_last >= v_idx, it's useful. Score = u_last (we evict the one we need least soon? No, we evict the one we need LATEST).
                    # Wait. FNU: Evict the one with the FURTHEST next use.
                    # So if u_last is large, it's good to evict? No.
                    # If u_last is large, it means we need it far in the future.
                    # If we evict it, we will have to load it later.
                    # If we keep it, we save a load later.
                    # So we should KEEP the ones with FURTHEST next use?
                    # NO. FNU says: Evict the page whose next use is FURTHEST in the future.
                    # Why? Because the pages whose next use is SOON will be used before we need to evict again.
                    # So we evict the one we are LEAST likely to need in the immediate future.
                    
                    # So, we want to evict u with MAX(u_last).
                    
                    score = u_last
                    if is_parent_of_v:
                        score -= 500 # Penalty for evicting a current parent
                    
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)

                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)
                    evicted = True

                if not evicted:
                    # Emergency: if budget is 0 or something weird
                    continue

            load(p)

        # Compute v
        if v not in in_fast:
            compute(v)
        else:
            # v was already in fast? Maybe from a previous computation?
            # In a DAG, we compute each node once in topo order.
            # But if we evicted and reloaded? No, we don't reload computed nodes unless we evicted them.
            # If v is in in_fast, it means it was computed earlier? 
            # In a strict topo order, v is computed here.
            # If v is in in_fast, it might be an input? No, inputs are not computed.
            # If v is in in_fast, we just skip compute?
            # But we must ensure it's the correct value.
            # Assuming topo order, v is not computed yet.
            # So v should NOT be in in_fast unless it's an input.
            # If v is an input, we don't compute it.
            if v in inputs:
                pass # It's an input, already in fast (loaded above)
            else:
                # This case shouldn't happen in a standard topo pass if we only add to in_fast via load/compute.
                # But if we loaded v as a parent of some other node? No, v is the current node.
                # So if v is in in_fast, it must have been loaded as a parent? 
                # But we are processing v in topo order. v's children come later.
                # So v cannot be a parent of a node processed earlier.
                # So v should not be in in_fast.
                # Unless the DAG has cycles? No, it's a DAG.
                # So this branch is likely dead code or handles re-computation?
                # The prompt says "Recomputations allowed".
                # If we recomputed v earlier, it's in in_fast.
                # We can just skip compute.
                pass

        # Clean up: store/delete vertices no longer needed
        # This is optional but helps keep memory clean.
        # We only do this if we have space or if we want to proactively free space.
        # Actually, proactive eviction is handled in the load step.
        # We can skip this to save operations (delete is not a transfer, but it's an op).
        # However, keeping useless items in fast memory prevents loading new ones.
        # The load step handles eviction. So we don't need proactive cleanup.

    # Ensure all outputs are in fast memory at the end
    for v in outputs:
        if v in needed_set and v not in in_fast:
            if len(in_fast) >= fast_memory_budget:
                # Evict something
                best_evict = None
                best_score = -1
                best_is_slow = False
                for u in list(in_fast):
                    if u in outputs: continue # Don't evict other outputs
                    u_last = last_child_idx.get(u, -1)
                    score = u_last
                    if score > best_score:
                        best_score = score
                        best_evict = u
                        best_is_slow = (u in in_slow)
                
                if best_evict is not None:
                    if best_is_slow:
                        delete(best_evict)
                    else:
                        store(best_evict)

            load(v)

    return operations
# EVOLVE-BLOCK-END
