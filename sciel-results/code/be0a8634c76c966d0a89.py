import sys
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = 80.0
    
    vertices = dag.get('vertices', [])
    edges = dag.get('edges', [])
    inputs = set(dag.get('inputs', []))
    outputs = set(dag.get('outputs', []))
    
    if not vertices:
        return []
        
    children = defaultdict(list)
    parents = defaultdict(list)
    for u, v in edges:
        children[u].append(v)
        parents[v].append(u)
        
    in_degree = {v: len(parents[v]) for v in vertices}
    queue = [v for v in vertices if in_degree[v] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != len(vertices):
        return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)

    # DP with parent tracking
    # dp[state] = (cost, prev_state, v, ops_delta)
    # We need to map (step, state) to parent info to reconstruct.
    # Since state is a frozenset, it's hashable.
    
    initial_state = frozenset()
    # (cost, prev_state, v, ops_delta)
    dp = {initial_state: (0, None, None, [])}
    
    processed = set()
    
    for v in topo_order:
        new_dp = {}
        if time.time() > deadline - 10:
            return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)
            
        for state, (cost, prev_state, prev_v, prev_ops) in dp.items():
            current_fast = set(state)
            
            # 1. Load parents
            load_ops = []
            for p in parents[v]:
                if p not in current_fast:
                    cost += 1
                    current_fast.add(p)
                    load_ops.append(('load', p))
            
            # 2. Compute v
            compute_op = ('compute', v)
            current_fast.add(v)
            
            # 3. Identify done nodes
            done_nodes = []
            for u in current_fast:
                if u == v:
                    continue
                all_children_processed = True
                for c in children[u]:
                    if c not in processed and c != v:
                        all_children_processed = False
                        break
                if all_children_processed:
                    done_nodes.append(u)
            
            # 4. Store done nodes
            store_ops = []
            for u in done_nodes:
                cost += 1
                current_fast.remove(u)
                store_ops.append(('store', u))
                
            # 5. Check budget
            if len(current_fast) > fast_memory_budget:
                continue
                
            new_state = frozenset(current_fast)
            
            if new_state not in new_dp or new_dp[new_state][0] > cost:
                new_dp[new_state] = (cost, state, v, load_ops + [compute_op] + store_ops)
                
        dp = new_dp
        processed.add(v)
        
        if not dp:
            return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)

    # Find best final state
    best_state = None
    best_cost = float('inf')
    for state, (cost, _, _, _) in dp.items():
        if cost < best_cost:
            best_cost = cost
            best_state = state
            
    if best_state is None:
        return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)
        
    # Reconstruct
    # We need to trace back from best_state.
    # But we overwrote dp. We need to store the history.
    # Let's modify the loop to store `history[step][state] = (prev_state, v, ops_delta)`
    
    # Re-run DP with history
    initial_state = frozenset()
    dp = {initial_state: (0, None, None, [])}
    history = [{}] # history[step][state] = (prev_state, v, ops_delta)
    processed = set()
    
    for v in topo_order:
        new_dp = {}
        new_history = {}
        if time.time() > deadline - 10:
            return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)
            
        for state, (cost, prev_state, prev_v, prev_ops) in dp.items():
            current_fast = set(state)
            
            load_ops = []
            for p in parents[v]:
                if p not in current_fast:
                    cost += 1
                    current_fast.add(p)
                    load_ops.append(('load', p))
            
            compute_op = ('compute', v)
            current_fast.add(v)
            
            done_nodes = []
            for u in current_fast:
                if u == v:
                    continue
                all_children_processed = True
                for c in children[u]:
                    if c not in processed and c != v:
                        all_children_processed = False
                        break
                if all_children_processed:
                    done_nodes.append(u)
            
            store_ops = []
            for u in done_nodes:
                cost += 1
                current_fast.remove(u)
                store_ops.append(('store', u))
                
            if len(current_fast) > fast_memory_budget:
                continue
                
            new_state = frozenset(current_fast)
            
            if new_state not in new_dp or new_dp[new_state][0] > cost:
                new_dp[new_state] = (cost, state, v, load_ops + [compute_op] + store_ops)
                new_history[new_state] = (state, v, load_ops + [compute_op] + store_ops)
                
        dp = new_dp
        history.append(new_history)
        processed.add(v)
        
        if not dp:
            return greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget)

    # Reconstruct from best_state
    current_state = best_state
    ops = []
    for i in range(len(topo_order) - 1, -1, -1):
        prev_state, v, ops_delta = history[i][current_state]
        ops.extend(reversed(ops_delta))
        current_state = prev_state
        
    return ops

def greedy_pebble(topo_order, parents, children, inputs, outputs, fast_memory_budget):
    fast_memory = set()
    ops = []
    processed = set()
    unprocessed_children = {v: len(children[v]) for v in topo_order}
    
    for v in topo_order:
        for p in parents[v]:
            if p not in fast_memory:
                ops.append(('load', p))
                fast_memory.add(p)
                
        ops.append(('compute', v))
        fast_memory.add(v)
        
        for p in parents[v]:
            unprocessed_children[p] -= 1
            if unprocessed_children[p] == 0:
                if p in fast_memory:
                    ops.append(('store', p))
                    fast_memory.remove(p)
                    
        while len(fast_memory) > fast_memory_budget:
            min_count = float('inf')
            evict_candidate = None
            for u in fast_memory:
                if unprocessed_children[u] < min_count:
                    min_count = unprocessed_children[u]
                    evict_candidate = u
            
            if evict_candidate is None:
                break
                
            ops.append(('store', evict_candidate))
            fast_memory.remove(evict_candidate)
            
        processed.add(v)
        
    for v in fast_memory:
        ops.append(('store', v))
        fast_memory.remove(v)
        
    return ops
