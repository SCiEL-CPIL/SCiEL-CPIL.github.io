import time
import random
import itertools

# Precomputation for n=20
N = 20
# Generate all edges
_all_edges = []
_edge_index = {}
for u in range(N):
    for v in range(u + 1, N):
        idx = len(_all_edges)
        _all_edges.append((u, v))
        _edge_index[(u, v)] = idx
        _edge_index[(v, u)] = idx

E = len(_all_edges) # 190

# Generate all 4-cycles
# A 4-cycle is defined by 4 distinct vertices.
# For 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles (up to rotation/reflection).
# 1: a-b-c-d-a
# 2: a-b-d-c-a
# 3: a-c-b-d-a
# We need to map these to edge indices.

_cycles = []
_edge_to_cycles = [[] for _ in range(E)]

def get_edge_idx(u, v):
    return _edge_index[(u, v)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, c)
    e3 = get_edge_idx(c, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 2: a-b-d-c-a
    e1 = get_edge_idx(a, b)
    e2 = get_edge_idx(b, d)
    e3 = get_edge_idx(d, c)
    e4 = get_edge_idx(c, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)
    
    # Cycle 3: a-c-b-d-a
    e1 = get_edge_idx(a, c)
    e2 = get_edge_idx(c, b)
    e3 = get_edge_idx(b, d)
    e4 = get_edge_idx(d, a)
    cid = len(_cycles)
    _cycles.append((e1, e2, e3, e4))
    _edge_to_cycles[e1].append(cid)
    _edge_to_cycles[e2].append(cid)
    _edge_to_cycles[e3].append(cid)
    _edge_to_cycles[e4].append(cid)

C = len(_cycles)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initialize colors randomly
    colors = [random.randint(0, 19) for _ in range(E)]
    
    # is_bad[cid] = True if the 4 edges in cycle cid do not have distinct colors
    is_bad = [False] * C
    bad_count = 0
    
    def check_cycle(cid):
        e1, e2, e3, e4 = _cycles[cid]
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        # Check if all distinct
        if len({c1, c2, c3, c4}) == 4:
            return False
        return True

    # Initial computation
    for cid in range(C):
        if check_cycle(cid):
            is_bad[cid] = True
            bad_count += 1
            
    # Simulated Annealing
    T = 1.0
    cooling = 0.99999
    min_T = 0.01
    
    # If already perfect, return
    if bad_count == 0:
        return [[u, v, colors[i]] for i, (u, v) in enumerate(_all_edges)]
        
    while time.time() < deadline - 5:  # Leave 5s for cleanup
        e_idx = random.randint(0, E - 1)
        old_c = colors[e_idx]
        
        # Pick a new color different from old_c
        new_c = random.randint(0, 19)
        while new_c == old_c:
            new_c = random.randint(0, 19)
            
        # Calculate delta
        delta = 0
        for cid in _edge_to_cycles[e_idx]:
            was_bad = is_bad[cid]
            # Simulate change
            e1, e2, e3, e4 = _cycles[cid]
            c1 = colors[e1] if e1 != e_idx else new_c
            c2 = colors[e2] if e2 != e_idx else new_c
            c3 = colors[e3] if e3 != e_idx else new_c
            c4 = colors[e4] if e4 != e_idx else new_c
            
            is_bad_now = (len({c1, c2, c3, c4}) < 4)
            
            if was_bad and not is_bad_now:
                delta -= 1
            elif not was_bad and is_bad_now:
                delta += 1
            
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            # Avoid division by zero or overflow
            if T < 1e-9:
                accept = False
            else:
                accept = random.random() < pow(2.718281828, -delta / T)
                
        if accept:
            colors[e_idx] = new_c
            # Update is_bad and bad_count
            for cid in _edge_to_cycles[e_idx]:
                new_state = check_cycle(cid)
                if new_state != is_bad[cid]:
                    if new_state:
                        bad_count += 1
                    else:
                        bad_count -= 1
                    is_bad[cid] = new_state
                    
            if bad_count == 0:
                break
                
        T *= cooling
        if T < min_T:
            T = min_T

    # If bad_count is 0, we are done
    if bad_count == 0:
        return [[u, v, colors[i]] for i, (u, v) in enumerate(_all_edges)]
        
    # Greedy Edge Removal
    # We need to remove edges to eliminate all bad cycles.
    # We maintain a set of active edges.
    active_edges = set(range(E))
    
    # List of bad cycle IDs
    bad_cycles = [cid for cid in range(C) if is_bad[cid]]
    
    # Map edge -> list of bad cycles it is in
    # This needs to be updated as we remove edges/cycles
    # To keep it simple and fast, we just re-scan bad_cycles each time if the list is small.
    # If the list is large, we might be in trouble, but SA should have reduced it.
    
    while bad_cycles and time.time() < deadline - 1:
        # Count frequency of each edge in bad_cycles
        edge_counts = {}
        for cid in bad_cycles:
            for e in _cycles[cid]:
                if e in active_edges:
                    edge_counts[e] = edge_counts.get(e, 0) + 1
                else:
                    # If an edge is already removed, the cycle is technically broken?
                    # No, a cycle exists only if all 4 edges exist.
                    # So if we remove an edge, all cycles containing it are no longer "cycles" in the graph.
                    # Thus, we should remove them from bad_cycles.
                    pass
        
        if not edge_counts:
            # All bad cycles contain at least one removed edge?
            # This shouldn't happen if we manage bad_cycles correctly.
            # But if we just removed an edge, we should have filtered bad_cycles.
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e, count in edge_counts.items():
            if count > max_count:
                max_count = count
                best_edge = e
                
        if best_edge == -1:
            break
            
        # Remove best_edge
        active_edges.remove(best_edge)
        
        # Update bad_cycles: remove any cycle containing best_edge
        new_bad_cycles = []
        for cid in bad_cycles:
            if best_edge in _cycles[cid]:
                # Cycle is broken
                pass
            else:
                new_bad_cycles.append(cid)
        bad_cycles = new_bad_cycles

    # Construct result from active_edges
    result = []
    for e in active_edges:
        u, v = _all_edges[e]
        result.append([u, v, colors[e]])
        
    return result

# EVOLVE-BLOCK-END
