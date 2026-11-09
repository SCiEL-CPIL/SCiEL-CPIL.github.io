"""Rewritten program for communication-minimizing DAG pebbling.
Implements a time-bounded search for a valid pebbling schedule that minimizes
transfers (load+store operations), with a deadline guard and multiple strategies.
"""
# EVOLVE-BLOCK-START
import time
import heapq
import math
from collections import defaultdict, deque

# Global deadline guard
_DEADLINE = None
def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 85.0  # 85s wall-clock limit (5s safety margin)
    return _DEADLINE

def _time_up():
    return time.time() >= _DEADLINE

def _topo_layers(dag):
    """Compute topological layers: list of lists of vertices by depth."""
    parents = dag.get('parents', {})
    children = defaultdict(list)
    for v, ps in parents.items():
        for p in ps:
            children[p].append(v)
    outputs = dag.get('outputs', [])
    inputs = [v for v in parents if v not in children]
    
    # BFS from inputs to assign layers
    layer = {}
    queue = deque()
    for v in inputs:
        layer[v] = 0
        queue.append(v)
    while queue:
        v = queue.popleft()
        for c in children.get(v, []):
            new_layer = layer[v] + 1
            if c not in layer or layer[c] < new_layer:
                layer[c] = new_layer
                queue.append(c)
    # Any vertex not reached (shouldn't happen)
    all_v = set(parents.keys()) | set(outputs)
    for v in all_v:
        if v not in layer:
            layer[v] = 0
    
    max_layer = max(layer.values()) if layer else 0
    layers = [[] for _ in range(max_layer + 1)]
    for v, l in layer.items():
        layers[l].append(v)
    return layers, inputs, outputs

def _count_transfers(schedule):
    """Count load+store operations in a schedule."""
    return sum(1 for op, v in schedule if op in ('load', 'store'))

def _verify_schedule(dag, schedule, fast_memory_budget):
    """Verify a pebbling schedule is valid. Returns True if valid."""
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    # State: vertex -> 'blue', 'red', or None
    state = {}
    fast_mem = 0  # count of red values in fast memory
    
    for op, v in schedule:
        if op == 'load':
            if state.get(v) == 'blue':
                continue  # already blue
            state[v] = 'red'
            fast_mem += 1
            if fast_mem > fast_memory_budget:
                return False
        elif op == 'store':
            if state.get(v) != 'red':
                return False
            state[v] = 'blue'
            fast_mem -= 1
        elif op == 'compute':
            # All parents must be red
            ps = parents.get(v, [])
            for p in ps:
                if state.get(p) != 'red':
                    return False
            # v becomes red (if not already)
            if state.get(v) == 'red':
                pass  # already red
            elif state.get(v) == 'blue':
                # Recompute: was blue, now red
                fast_mem += 1
            else:
                fast_mem += 1
            state[v] = 'red'
            if fast_mem > fast_memory_budget:
                return False
        elif op == 'delete':
            if state.get(v) != 'red':
                return False
            del state[v]
            fast_mem -= 1
        else:
            return False
    
    # All outputs must be blue
    for v in outputs:
        if state.get(v) != 'blue':
            return False
    return True

def _greedy_layered(dag, fast_memory_budget):
    """Simple layered greedy: process layer by layer, load inputs, compute, store outputs."""
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    layers, inputs, outputs_list = _topo_layers(dag)
    
    schedule = []
    state = {}  # vertex -> 'blue', 'red', None
    fast_mem = 0
    
    for layer in layers:
        # First, ensure all vertices in this layer that need to be computed have parents ready
        # Load any inputs in this layer that aren't blue
        for v in layer:
            if v in inputs and state.get(v) != 'blue':
                state[v] = 'red'
                fast_mem += 1
                schedule.append(('load', v))
        
        # Compute vertices in this layer (those with parents)
        for v in layer:
            ps = parents.get(v, [])
            if not ps:
                continue  # input, already loaded
            # Check parents are red
            all_red = all(state.get(p) == 'red' for p in ps)
            if all_red:
                # Compute v
                if state.get(v) != 'red':
                    fast_mem += 1
                state[v] = 'red'
                schedule.append(('compute', v))
        
        # Store vertices that are outputs or will be needed later
        for v in layer:
            if v in outputs_list and state.get(v) == 'red':
                state[v] = 'blue'
                fast_mem -= 1
                schedule.append(('store', v))
    
    # Store any remaining red outputs
    for v in outputs_list:
        if state.get(v) == 'red':
            state[v] = 'blue'
            fast_mem -= 1
            schedule.append(('store', v))
    
    return schedule

def _naive_schedule(dag, fast_memory_budget):
    """Naive schedule: load all inputs, compute in topo order, store outputs."""
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    layers, inputs, outputs_list = _topo_layers(dag)
    
    schedule = []
    # Load all inputs
    for v in inputs:
        schedule.append(('load', v))
    
    # Compute in topological order
    for layer in layers:
        for v in layer:
            ps = parents.get(v, [])
            if ps:
                schedule.append(('compute', v))
    
    # Store all outputs
    for v in outputs_list:
        schedule.append(('store', v))
    
    return schedule

def _improved_greedy(dag, fast_memory_budget):
    """Improved greedy with better memory management."""
    parents = dag.get('parents', {})
    outputs = set(dag.get('outputs', []))
    layers, inputs, outputs_list = _topo_layers(dag)
    
    schedule = []
    state = {}  # vertex -> 'blue', 'red', None
    fast_mem = 0
    remaining_children = defaultdict(int)
    
    # Count how many children each vertex has
    for v, ps in parents.items():
        for p in ps:
            remaining_children[p] += 1
    
    for layer_idx, layer in enumerate(layers):
        # Load inputs in this layer
        for v in layer:
            if v in inputs and state.get(v) is None:
                state[v] = 'red'
                fast_mem += 1
                schedule.append(('load', v))
        
        # Compute vertices in this layer
        for v in layer:
            ps = parents.get(v, [])
            if not ps:
                continue
            all_red = all(state.get(p) == 'red' for p in ps)
            if all_red:
                if state.get(v) != 'red':
                    fast_mem += 1
                state[v] = 'red'
                schedule.append(('compute', v))
        
        # Store vertices that are outputs or have no more children needing them
        for v in layer:
            if state.get(v) != 'red':
                continue
            # Store if it's an output
            if v in outputs:
                state[v] = 'blue'
                fast_mem -= 1
                schedule.append(('store', v))
            # Or if fast memory is tight and we can store
            elif fast_mem >= fast_memory_budget and remaining_children[v] == 0:
                state[v] = 'blue'
                fast_mem -= 1
                schedule.append(('store', v))
    
    # Final: ensure all outputs are blue
    for v in outputs_list:
        if state.get(v) == 'red':
            state[v] = 'blue'
            fast_mem -= 1
            schedule.append(('store', v))
    
    return schedule

def _local_search(dag, fast_memory_budget, initial_schedule, max_iters=200):
    """Local search to improve schedule by swapping/reordering operations."""
    if len(initial_schedule) > 20000:
        return initial_schedule
    
    best = initial_schedule[:]
    best_cost = _count_transfers(best)
    
    # Try simple improvements: remove redundant operations
    for it in range(max_iters):
        if _time_up():
            break
        improved = False
        new_schedule = best[:]
        
        # Try removing each operation and checking validity
        for i in range(len(new_schedule)):
            if _time_up():
                break
            candidate = new_schedule[:i] + new_schedule[i+1:]
            if len(candidate) < len(best) and _verify_schedule(dag, candidate, fast_memory_budget):
                cost = _count_transfers(candidate)
                if cost < best_cost:
                    best = candidate
                    best_cost = cost
                    improved = True
                    new_schedule = candidate
                    break
        if not improved:
            break
    
    return best

def pebble(dag, fast_memory_budget):
    """Main entry point: find a valid pebbling schedule minimizing transfers."""
    _init_deadline()
    
    parents = dag.get('parents', {})
    outputs = dag.get('outputs', [])
    
    # Generate candidate schedules
    candidates = []
    
    # 1. Naive schedule
    try:
        naive = _naive_schedule(dag, fast_memory_budget)
        if _verify_schedule(dag, naive, fast_memory_budget) and len(naive) <= 20000:
            candidates.append(naive)
    except Exception:
        pass
    
    # 2. Greedy layered
    try:
        greedy = _greedy_layered(dag, fast_memory_budget)
        if _verify_schedule(dag, greedy, fast_memory_budget) and len(greedy) <= 20000:
            candidates.append(greedy)
    except Exception:
        pass
    
    # 3. Improved greedy
    try:
        improved = _improved_greedy(dag, fast_memory_budget)
        if _verify_schedule(dag, improved, fast_memory_budget) and len(improved) <= 20000:
            candidates.append(improved)
    except Exception:
        pass
    
    if not candidates:
        # Fallback: minimal valid schedule
        return _naive_schedule(dag, fast_memory_budget)
    
    # Pick the best candidate by transfer count
    best = min(candidates, key=_count_transfers)
    
    # Try local search improvement
    if not _time_up():
        try:
            improved = _local_search(dag, fast_memory_budget, best, max_iters=50)
            if _verify_schedule(dag, improved, fast_memory_budget):
                best = improved
        except Exception:
            pass
    
    # Ensure we don't exceed operation limit
    if len(best) > 20000:
        best = best[:20000]
        # Verify truncated; if invalid, fall back to naive
        if not _verify_schedule(dag, best, fast_memory_budget):
            best = _naive_schedule(dag, fast_memory_budget)
            if len(best) > 20000:
                best = best[:20000]
    
    return best
# EVOLVE-BLOCK-END
