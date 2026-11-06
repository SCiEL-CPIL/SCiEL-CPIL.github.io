import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N = len(ALL_TRIPLES)  # 560

# Precompute all K(2,2,2) patterns: each is a set of 8 triples on 6 distinct vertices
# For each 6-set {a,a',b,b',c,c'}, the 8 transversals are:
# (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
# We need to enumerate all ways to partition 6 vertices into 3 pairs and then the 8 transversals.

# Actually, a K(2,2,2) is defined by 3 disjoint pairs (A,A'), (B,B'), (C,C').
# The 8 transversals pick one from each pair: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'),
# (a',b,c), (a',b,c'), (a',b',c), (a',b',c')

# Precompute all K(2,2,2) instances: list of frozensets of 8 triple indices
# This is expensive but done once.

def _precompute_k222():
    """Precompute all K(2,2,2) patterns as sets of triple indices."""
    # Map from triple (a,b,c) with a<b<c to index
    triple_to_idx = {}
    for i, t in enumerate(ALL_TRIPLES):
        triple_to_idx[t] = i
    
    k222_patterns = []
    
    # Choose 6 vertices from 16
    for verts in combinations(range(V), 6):
        # Choose a partition into 3 pairs: C(6,2)*C(4,2)*C(2,2)/3! = 15*6*1/6 = 15
        # Actually: number of ways to partition 6 elements into 3 pairs = 6!/(2!^3 * 3!) = 720/(8*6) = 15
        v0, v1, v2, v3, v4, v5 = verts
        # Generate all perfect matchings of 6 vertices
        # Pair (v0, vi) for i in 1..5, then pair remaining 4
        for i in range(1, 6):
            # First pair: (v0, vi)
            remaining = [v for j, v in enumerate(verts) if j != 0 and j != i]
            # Pair remaining 4: 3 ways
            r0, r1, r2, r3 = remaining
            # Pairing 1: (r0,r1),(r2,r3)
            pairs1 = [(v0, vi), (r0, r1), (r2, r3)]
            # Pairing 2: (r0,r2),(r1,r3)
            pairs2 = [(v0, vi), (r0, r2), (r1, r3)]
            # Pairing 3: (r0,r3),(r1,r2)
            pairs3 = [(v0, vi), (r0, r3), (r1, r2)]
            
            for pairs in [pairs1, pairs2, pairs3]:
                # Sort pairs to normalize
                pairs = sorted(pairs, key=lambda p: min(p))
                a, a2 = pairs[0]
                b, b2 = pairs[1]
                c, c2 = pairs[2]
                
                # Generate 8 transversals
                transversals = []
                for x in [a, a2]:
                    for y in [b, b2]:
                        for z in [c, c2]:
                            t = tuple(sorted((x, y, z)))
                            transversals.append(t)
                
                # Check all are valid triples
                if all(t in triple_to_idx for t in transversals):
                    idx_set = frozenset(triple_to_idx[t] for t in transversals)
                    if len(idx_set) == 8:
                        k222_patterns.append(idx_set)
    
    return k222_patterns

_K222 = _precompute_k222()
_N_K222 = len(_K222)

# For each triple, list of K(2,2,2) patterns it belongs to
_TRIPLE_TO_K222 = [[] for _ in range(N)]
for k_idx, pattern in enumerate(_K222):
    for t_idx in pattern:
        _TRIPLE_TO_K222[t_idx].append(k_idx)


def _check_conflict_fast(triple_idx, presence):
    """Check if adding triple triple_idx creates a K(2,2,2)."""
    for k_idx in _TRIPLE_TO_K222[triple_idx]:
        pattern = _K222[k_idx]
        # Check if all other 7 triples in this pattern are present
        all_present = True
        for t in pattern:
            if t != triple_idx and not presence[t]:
                all_present = False
                break
        if all_present:
            return True
    return False


def _check_conflict_count(triple_idx, presence):
    """Check if adding triple triple_idx creates a K(2,2,2). Returns True if conflict."""
    for k_idx in _TRIPLE_TO_K222[triple_idx]:
        pattern = _K222[k_idx]
        count = 0
        for t in pattern:
            if t != triple_idx and presence[t]:
                count += 1
        if count == 7:
            return True
    return False


def _greedy_construct(order, presence, deadline):
    """Add triples in given order, skipping conflicts."""
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        if _check_conflict_fast(idx, presence):
            continue
        presence[idx] = True
        count += 1
    return count


def _local_search(presence, count, deadline):
    """
    Local search:
    1. Try to add any missing triple
    2. Try 1-swap: remove one present, add as many missing as possible
    3. Try 2-swap: remove two present, add as many missing as possible
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: add any possible missing triple
        for idx in range(N):
            if time.time() >= deadline:
                return count, presence
            if presence[idx]:
                continue
            if not _check_conflict_fast(idx, presence):
                presence[idx] = True
                count += 1
                improved = True
        
        if improved:
            continue
        
        # Phase 2: try 1-swaps
        present_list = [i for i in range(N) if presence[i]]
        if not present_list:
            break
        
        random.shuffle(present_list)
        max_swaps = min(50, len(present_list))
        
        for idx_out in present_list[:max_swaps]:
            if time.time() >= deadline:
                return count, presence
            
            # Save state
            pres_backup = presence[:]
            count_backup = count
            
            presence[idx_out] = False
            count -= 1
            
            # Try to add all possible missing triples
            for idx_in in range(N):
                if time.time() >= deadline:
                    presence[:] = pres_backup
                    count = count_backup
                    return count, presence
                if presence[idx_in]:
                    continue
                if not _check_conflict_fast(idx_in, presence):
                    presence[idx_in] = True
                    count += 1
            
            if count > count_backup:
                improved = True
                break
            else:
                presence[:] = pres_backup
                count = count_backup
        
        if improved:
            continue
        
        # Phase 3: try 2-swaps (remove 2, add many)
        present_list = [i for i in range(N) if presence[i]]
        if len(present_list) < 2:
            break
        
        random.shuffle(present_list)
        max_swaps2 = min(20, len(present_list) // 2)
        
        for i in range(max_swaps2):
            if time.time() >= deadline:
                return count, presence
            
            idx_out1 = present_list[i * 2]
            idx_out2 = present_list[i * 2 + 1]
            
            pres_backup = presence[:]
            count_backup = count
            
            presence[idx_out1] = False
            presence[idx_out2] = False
            count -= 2
            
            # Try to add all possible missing triples
            for idx_in in range(N):
                if time.time() >= deadline:
                    presence[:] = pres_backup
                    count = count_backup
                    return count, presence
                if presence[idx_in]:
                    continue
                if not _check_conflict_fast(idx_in, presence):
                    presence[idx_in] = True
                    count += 1
            
            if count > count_backup:
                improved = True
                break
            else:
                presence[:] = pres_backup
                count = count_backup
    
    return count, presence


def _simulated_annealing(presence, count, deadline, start_temp=10.0, end_temp=0.1):
    """
    Simulated annealing:
    - Move: add a random missing triple, or remove a random present triple, or swap
    - Accept if improves, or with probability exp(-delta/T) if worsens
    """
    current_count = count
    current_presence = presence[:]
    best_count = count
    best_presence = presence[:]
    
    T = start_temp
    n_steps = 0
    max_steps = 50000
    cool_rate = (end_temp / start_temp) ** (1.0 / max_steps)
    
    while time.time() < deadline and n_steps < max_steps:
        n_steps += 1
        T *= cool_rate
        
        # Random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random missing triple
            missing = [i for i in range(N) if not current_presence[i]]
            if not missing:
                continue
            idx = random.choice(missing)
            if _check_conflict_fast(idx, current_presence):
                continue
            current_presence[idx] = True
            current_count += 1
            delta = 1
        elif move_type < 0.5:
            # Remove a random present triple
            present = [i for i in range(N) if current_presence[i]]
            if not present:
                continue
            idx = random.choice(present)
            current_presence[idx] = False
            current_count -= 1
            delta = -1
        else:
            # Swap: remove one present, add one missing
            present = [i for i in range(N) if current_presence[i]]
            missing = [i for i in range(N) if not current_presence[i]]
            if not present or not missing:
                continue
            idx_out = random.choice(present)
            idx_in = random.choice(missing)
            
            # Check if idx_in can be added after removing idx_out
            current_presence[idx_out] = False
            current_count -= 1
            
            if _check_conflict_fast(idx_in, current_presence):
                # Revert
                current_presence[idx_out] = True
                current_count += 1
                continue
            
            current_presence[idx_in] = True
            current_count += 1
            delta = 0  # Net zero, but may enable future additions
        
        # Accept/reject
        if delta >= 0:
            if current_count > best_count:
                best_count = current_count
                best_presence = current_presence[:]
        else:
            # Worsening move, accept with probability
            if random.random() < (2.718281828 ** (-delta / max(T, 1e-10))):
                pass  # Accept
            else:
                # Revert
                if move_type < 0.5:
                    # Was a removal
                    current_presence[idx_out] = True
                    current_count += 1
                else:
                    # Was a swap
                    current_presence[idx_in] = False
                    current_presence[idx_out] = True
                    current_count -= 1
    
    return best_count, best_presence


def construct():
    start = time.time()
    deadline = start + 115  # safety margin for 150s limit
    
    best_count = 0
    best_presence = [False] * N
    
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        presence = [False] * N
        
        # Strategy 1: random order greedy
        if attempts % 3 == 1:
            order = list(range(N))
            random.shuffle(order)
            count = _greedy_construct(order, presence, deadline)
            count, presence = _local_search(presence, count, deadline)
        
        # Strategy 2: simulated annealing from random start
        elif attempts % 3 == 2:
            # Start with a small random set
            for _ in range(50):
                idx = random.randint(0, N - 1)
                if not presence[idx] and not _check_conflict_fast(idx, presence):
                    presence[idx] = True
            count = sum(presence)
            count, presence = _simulated_annealing(presence, count, deadline, start_temp=5.0, end_temp=0.5)
        
        # Strategy 3: greedy + local search + SA
        else:
            order = list(range(N))
            random.shuffle(order)
            count = _greedy_construct(order, presence, deadline)
            count, presence = _local_search(presence, count, deadline)
            count, presence = _simulated_annealing(presence, count, deadline, start_temp=2.0, end_temp=0.2)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
    
    result = []
    for i in range(N):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result

random.seed(42)
