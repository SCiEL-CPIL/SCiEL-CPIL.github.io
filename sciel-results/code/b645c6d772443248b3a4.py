import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0

def _construct_neighbors():
    """
    Precompute neighbors for each node 1..256.
    i and j are neighbors if (i*j) % (i+j) == 0.
    """
    neighbors = [set() for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors

def _can_add(x, S_set, neighbors):
    """Check if x can be added to S_set."""
    return not (neighbors[x] & S_set)

def _spectral_init(neighbors, rng):
    """
    Spectral clustering approach: compute eigenvector of conflict graph,
    use it to identify a low-conflict core, then greedily expand.
    """
    # Build adjacency matrix of the conflict graph
    A = np.zeros((_N + 1, _N + 1))
    for i in range(1, _N + 1):
        for j in neighbors[i]:
            A[i, j] = 1
            A[j, i] = 1
    
    # Compute largest eigenvector (principal component)
    # Use power iteration for efficiency
    v = np.ones(_N + 1)
    v[0] = 0
    v = v / np.linalg.norm(v)
    
    for _ in range(100):
        new_v = A @ v
        norm = np.linalg.norm(new_v)
        if norm < 1e-15:
            break
        v = new_v / norm
    
    # Eigenvector values: higher value = more "central" in conflict graph
    # For independent set, we want nodes with LOW eigenvector values (peripheral)
    # Sort by eigenvector value ascending (least conflicting first)
    order = list(range(1, _N + 1))
    order.sort(key=lambda x: v[x])
    
    # Greedy expansion using spectral ordering
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    
    return S, S_set

def _greedy_init(neighbors, rng, mode='degree'):
    """
    Greedy initialization.
    mode: 'degree', 'random', 'least_conflicting', 'spectral'
    """
    if mode == 'degree':
        degrees = [len(neighbors[i]) for i in range(_N + 1)]
        order = list(range(1, _N + 1))
        order.sort(key=lambda x: (degrees[x], x))
    elif mode == 'random':
        order = list(range(1, _N + 1))
        rng.shuffle(order)
    elif mode == 'spectral':
        S, S_set = _spectral_init(neighbors, rng)
        return S, S_set
    else:
        order = list(range(1, _N + 1))

    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _local_search(S, neighbors, rng, time_limit):
    """
    Local search to improve S.
    1. Try to add any available node.
    2. Try 1-for-k swaps (remove u, greedily refill).
    """
    S = list(S)
    S_set = set(S)
    complement = set(range(1, _N + 1)) - S_set
    start = time.time()
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        
        # Phase 1: Add any node that fits
        for x in list(complement):
            if _can_add(x, S_set, neighbors):
                S.append(x)
                S_set.add(x)
                complement.discard(x)
                improved = True
        
        if time.time() >= time_limit:
            break
            
        # Phase 2: Try swaps
        if not improved:
            best_S = S
            best_S_set = S_set
            best_size = len(S)
            
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                S_set.remove(u)
                complement.add(u)
                
                new_S_set = set(S_set)
                new_complement = set(complement)
                
                for x in list(new_complement):
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        new_complement.discard(x)
                        
                if len(new_S_set) > best_size:
                    best_S = list(new_S_set)
                    best_S_set = new_S_set
                    best_size = len(new_S_set)
                    improved = True
                    S = best_S
                    S_set = best_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    break 

    return S

def _simulated_annealing(S, neighbors, rng, time_limit):
    """
    Simulated annealing to escape local optima.
    Try random swaps and accept with probability based on temperature.
    """
    S = list(S)
    S_set = set(S)
    start = time.time()
    
    # Initial temperature
    T = 1.0
    cooling = 0.995
    
    best_S = list(S)
    best_size = len(S)
    
    while time.time() < time_limit and T > 0.01:
        # Pick a random element from S and a random element from complement
        S_list = list(S)
        complement = set(range(1, _N + 1)) - S_set
        
        if not S_list or not complement:
            break
            
        u = rng.choice(S_list)
        v = rng.choice(list(complement))
        
        # Try swap: remove u, add v
        if u == v:
            continue
            
        S_set.remove(u)
        S_set.add(v)
        
        # Check validity
        valid = True
        for x in S_set:
            if x != v and (x * v) % (x + v) == 0:
                valid = False
                break
        
        if valid:
            # New size is same (swap), but might enable further additions
            new_S = list(S_set)
            new_S_set = set(S_set)
            new_complement = set(range(1, _N + 1)) - new_S_set
            
            # Greedily add any available nodes
            for x in list(new_complement):
                if _can_add(x, new_S_set, neighbors):
                    new_S_set.add(x)
                    new_complement.discard(x)
            
            new_size = len(new_S_set)
            
            if new_size > best_size:
                best_S = list(new_S_set)
                best_size = new_size
                S = best_S
                S_set = best_S_set
            elif new_size == len(S):
                # Accept with probability exp(0/T) = 1 (same size)
                pass
            else:
                # Revert
                S_set.remove(v)
                S_set.add(u)
        else:
            # Invalid swap, revert
            S_set.remove(v)
            S_set.add(u)
        
        T *= cooling
    
    return best_S if len(best_S) > len(S) else S

def _two_swap_search(S, neighbors, rng, time_limit):
    """
    Try 2-for-3 swaps: remove two elements from S, add up to three from remaining.
    """
    S = list(S)
    S_set = set(S)
    remaining = set(range(1, _N + 1)) - S_set
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, n):
                if time.time() >= time_limit:
                    break
                r1, r2 = S_list[i], S_list[j]
                S_set.discard(r1)
                S_set.discard(r2)
                
                new_S_set = set(S_set)
                added = 0
                for x in list(remaining):
                    if added >= 3:
                        break
                    if _can_add(x, new_S_set, neighbors):
                        new_S_set.add(x)
                        added += 1
                
                if added > 2:
                    S = list(new_S_set)
                    S_set = new_S_set
                    remaining = set(range(1, _N + 1)) - S_set
                    improved = True
                    break
                else:
                    S_set.add(r1)
                    S_set.add(r2)
            if improved:
                break
    return S

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    neighbors = _construct_neighbors()
    
    best_S = []
    best_size = 0
    
    # Try multiple random restarts of greedy init
    for i in range(150):
        if time.time() - t0 > _DEADLINE - 10:
            break
        
        # Variation 1: Degree sorted
        if i % 5 == 0:
            S, S_set = _greedy_init(neighbors, rng, mode='degree')
        elif i % 5 == 1:
            S, S_set = _greedy_init(neighbors, rng, mode='random')
        elif i % 5 == 2:
            # Least conflicting greedy (dynamic)
            S, S_set = [], set()
            remaining = set(range(1, _N + 1))
            while remaining:
                best_x = None
                best_c = None
                for x in remaining:
                    c = len(neighbors[x] & S_set)
                    if best_c is None or c < best_c or (c == best_c and x < best_x):
                        best_x = x
                        best_c = c
                    if c == 0:
                        break
                if best_c == 0:
                    S.append(best_x)
                    S_set.add(best_x)
                    remaining.discard(best_x)
                else:
                    break
        elif i % 5 == 3:
            # Spectral init
            S, S_set = _greedy_init(neighbors, rng, mode='spectral')
        else:
            # Randomized spectral: perturb eigenvector ordering
            S, S_set = _greedy_init(neighbors, rng, mode='random')
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 5:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 5)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Two-swap search
    if time.time() - t0 < _DEADLINE - 3:
        S_final = _two_swap_search(best_S, neighbors, rng, _DEADLINE - 3)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
            
    # Simulated annealing polish
    if time.time() - t0 < _DEADLINE - 1:
        S_final = _simulated_annealing(best_S, neighbors, rng, _DEADLINE - 1)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
          
    return best_S
