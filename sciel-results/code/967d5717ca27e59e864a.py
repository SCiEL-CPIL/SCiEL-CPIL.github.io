import time
import random

_N = 256
_DEADLINE = 85.0

def _construct_neighbors():
    """
    Precompute neighbors for each node 1..256.
    i and j are neighbors if (i*j) % (i+j) == 0.
    """
    neighbors = [[] for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i + 1, _N + 1):
            if (i * j) % (i + j) == 0:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors

def _can_add(x, S_set, neighbors):
    """Check if x can be added to S_set."""
    for y in neighbors[x]:
        if y in S_set:
            return False
    return True

def _greedy_init(neighbors, rng):
    """
    Greedy initialization.
    Sort nodes by degree (ascending) to pick 'safe' ones first.
    """
    degrees = [len(neighbors[i]) for i in range(_N + 1)]
    order = list(range(1, _N + 1))
    order.sort(key=lambda x: (degrees[x], x))
    
    S = []
    S_set = set()
    for x in order:
        if _can_add(x, S_set, neighbors):
            S.append(x)
            S_set.add(x)
    return S, S_set

def _greedy_refill(neighbors, S_set, complement, rng=None):
    """
    Greedily add nodes from complement to S_set.
    Order candidates by number of neighbors in S_set (fewest first) to maximize additions.
    """
    new_S_set = set(S_set)
    remaining = set(complement)
    
    # Sort by conflict count with current S_set
    def conflict_count(x):
        return sum(1 for y in neighbors[x] if y in new_S_set)
    
    sorted_remaining = sorted(remaining, key=conflict_count)
    
    added = 0
    for x in sorted_remaining:
        if _can_add(x, new_S_set, neighbors):
            new_S_set.add(x)
            added += 1
    
    return new_S_set, added

def _local_search(S, neighbors, rng, time_limit):
    """
    Local search to improve S.
    1. Try to add any available node.
    2. Try 1-for-k swaps (remove u, greedily refill).
    3. Try 2-swap moves (remove u,v, greedily refill).
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
            
        # Phase 2: Try 1-for-k swaps if stuck
        if not improved:
            best_size = len(S)
            
            for u in list(S):
                if time.time() >= time_limit:
                    break
                
                # Remove u
                S_set.remove(u)
                complement.add(u)
                
                # Greedily refill using conflict-count ordering
                new_S_set, _ = _greedy_refill(neighbors, S_set, complement, rng)
                
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(new_S_set)
                    improved = True
                    break
            
            if improved:
                continue
        
        # Phase 3: Try 2-swap moves if still stuck
        if not improved and time.time() < time_limit:
            best_size = len(S)
            S_list = list(S)
            n = len(S_list)
            
            # Try pairs of removals, but limit to avoid O(n^2) blowup
            # Sample a subset of pairs if n is large
            max_pairs = 200
            if n > 30:
                # Random sample of pairs
                pair_indices = []
                for _ in range(max_pairs):
                    i, j = rng.sample(range(n), 2)
                    pair_indices.append((i, j))
            else:
                pair_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
            
            for i, j in pair_indices:
                if time.time() >= time_limit:
                    break
                
                u, v = S_list[i], S_list[j]
                
                # Remove u and v
                S_set.remove(u)
                S_set.remove(v)
                complement.add(u)
                complement.add(v)
                
                # Greedily refill
                new_S_set, _ = _greedy_refill(neighbors, S_set, complement, rng)
                
                if len(new_S_set) > best_size:
                    S = list(new_S_set)
                    S_set = new_S_set
                    complement = set(range(1, _N + 1)) - S_set
                    best_size = len(new_S_set)
                    improved = True
                    break

    return S

def _validate(S):
    """Check that S is a valid independent set."""
    S_set = set(S)
    for a in S:
        for b in S:
            if a < b and (a * b) % (a + b) == 0:
                return False
    return True

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    neighbors = _construct_neighbors()
    
    best_S = []
    best_size = 0
    
    # Try multiple random restarts of greedy init
    for i in range(50):
        if time.time() - t0 > _DEADLINE - 5:
            break
        
        if i % 2 == 0:
            S, S_set = _greedy_init(neighbors, rng)
        else:
            # Random order
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            S = []
            S_set = set()
            for x in order:
                if _can_add(x, S_set, neighbors):
                    S.append(x)
                    S_set.add(x)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Local search on best found
    if time.time() - t0 < _DEADLINE - 2:
        S_final = _local_search(best_S, neighbors, rng, _DEADLINE - 2)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)
    
    # Final validation safety net
    if not _validate(best_S):
        # Fallback: use all odd numbers (no two odds have even sum dividing product)
        fallback = list(range(1, 257, 2))
        if _validate(fallback):
            return fallback
        # Last resort: single element
        return [1]
    
    return best_S
