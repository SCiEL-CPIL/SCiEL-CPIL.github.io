"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _independence_number_fast(triples):
    """Compute independence number using bitmask superset marking."""
    n = 16
    # Mark all subsets that contain at least one triple
    # A subset S is independent iff no triple is a subset of S
    # i.e., S is independent iff S does NOT contain any triple
    # So we mark all supersets of each triple as "not independent"
    not_independent = [False] * (1 << n)
    for t in triples:
        triple_mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        # Mark all supersets of triple_mask
        # Supersets of mask m: m | x for all x subset of ~m
        complement = ((1 << n) - 1) ^ triple_mask
        sub = complement
        while True:
            not_independent[triple_mask | sub] = True
            if sub == 0:
                break
            sub = (sub - 1) & complement
    
    # Find largest independent set
    best = 0
    for mask in range(1 << n):
        if not not_independent[mask]:
            cnt = bin(mask).count('1')
            if cnt > best:
                best = cnt
    return best

def _is_valid_system(triples):
    """Check that any two triples share at most one point."""
    n = len(triples)
    for i in range(n):
        ti = set(triples[i])
        for j in range(i + 1, n):
            tj = set(triples[j])
            if len(ti & tj) > 1:
                return False
    return True

def _can_add(triples, candidate):
    """Check if adding candidate triple maintains linearity."""
    cand_set = set(candidate)
    for t in triples:
        if len(cand_set & set(t)) > 1:
            return False
    return True

def _algebraic_design():
    """
    Construct a linear triple system using the 4-bit vector space F2^4.
    Points are 0..15. Use affine lines (1D subspaces shifted) and 
    other structured triples to maximize coverage.
    
    In F2^4, an affine line through a in direction v is {a, a+v, a+2v} = {a, a+v, a} since 2v=0 in F2.
    Wait, in F2, 2v = 0, so a line has only 2 points. We need triples.
    
    Instead, use "planes" or other structures. A 2D subspace in F2^4 has 4 points.
    We can take triples from these.
    
    Alternative: Use the fact that in F2^4, we can define triples as:
    - For each 1D subspace direction v != 0, and each point a, the set {a, a+v, a+w} 
      where w is another direction... this gets complicated.
    
    Let me use a different approach: generate all affine 2-flats (4-point sets) and
    select triples from them carefully.
    
    Actually, let's use a simpler constructive approach:
    - Partition into 4 groups of 4 (the 2D subspaces of F2^4)
    - Within each group, add all C(4,3)=4 triples
    - Between groups, add structured triples
    
    F2^4 has 15 nonzero vectors. The 1D subspaces (lines through origin) are 7 (each pair of nonzero vectors in same 1D subspace... wait, in F2, each 1D subspace has exactly 2 nonzero elements? No, F2^1 has 2 elements: 0 and 1. So a 1D subspace of F2^4 spanned by v is {0, v}. There are (2^4 - 1)/(2-1) = 15 such subspaces.
    
    2D subspaces: Each has 4 elements. Number of 2D subspaces in F2^4: 
    Gaussian binomial [4 choose 2]_2 = (15*14)/(3*2) = 35.
    
    Let me just generate all 2D subspaces and use their 4-point sets to create triples.
    """
    points = list(range(16))
    
    # Generate all 2D subspaces of F2^4
    # A 2D subspace is spanned by two linearly independent vectors u, v
    # It contains {0, u, v, u+v}
    subspaces = set()
    for i in range(16):
        for j in range(16):
            if i == 0 or j == 0:
                continue
            # Check linear independence: i & j should not make one a multiple of the other
            # In F2, i and j are dependent iff i == j or i == 0 or j == 0
            # Actually in F2^4, two nonzero vectors are dependent iff they are equal
            if i == j:
                continue
            # The subspace is {0, i, j, i^j}
            subs = frozenset([0, i, j, i ^ j])
            # Normalize: use sorted tuple as key
            subspaces.add(subs)
    
    # Now build a linear triple system from these 4-point sets
    # From each 4-point set, we can add up to 4 triples (all C(4,3))
    # But we need to ensure linearity across all triples
    
    # Strategy: Use a greedy approach over the 2D subspaces
    # For each 4-point set, try to add triples from it
    
    all_triples = []
    used_pairs = set()  # Track pairs of points that appear together
    
    def add_triple(a, b, c):
        """Add triple if no pair is already used."""
        pairs = frozenset([(a, b), (a, c), (b, c)])
        for p in pairs:
            if p in used_pairs:
                return False
        for p in pairs:
            used_pairs.add(p)
        all_triples.append([a, b, c])
        return True
    
    # Process each 2D subspace
    for subs in sorted(subspaces, key=lambda s: tuple(sorted(s))):
        pts = sorted(subs)
        # Try all 4 triples from this 4-set
        for i in range(4):
            for j in range(i+1, 4):
                for k in range(j+1, 4):
                    add_triple(pts[i], pts[j], pts[k])
    
    return all_triples

def _greedy_build():
    """Build a linear triple system using greedy randomized approach."""
    random.shuffle(list(range(16)))
    all_triples = list(combinations(range(16), 3))
    random.shuffle(all_triples)
    
    system = []
    for t in all_triples:
        if _can_add(system, t):
            system.append(list(t))
    return system

def _local_search(triples, time_limit=10):
    """Improve the system by swapping triples to reduce independence number."""
    start = time.time()
    current_alpha = _independence_number_fast(triples)
    best_alpha = current_alpha
    best_system = [list(t) for t in triples]
    
    while time.time() - start < time_limit:
        improved = False
        random.shuffle(triples)
        
        for i in range(len(triples)):
            if time.time() - start > time_limit:
                return best_system, best_alpha
            
            remaining = triples[:i] + triples[i+1:]
            
            for attempt in range(30):
                candidate = list(random.sample(range(16), 3))
                if _can_add(remaining, candidate):
                    new_system = remaining + [candidate]
                    new_alpha = _independence_number_fast(new_system)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        triples = new_system
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [list(t) for t in triples]
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try 2-opt
            if len(triples) >= 2:
                i, j = random.sample(range(len(triples)), 2)
                i, j = min(i, j), max(i, j)
                remaining = triples[:i] + triples[i+1:j] + triples[j+1:]
                
                for attempt in range(40):
                    c1 = list(random.sample(range(16), 3))
                    c2 = list(random.sample(range(16), 3))
                    if _can_add(remaining, c1) and _can_add(remaining + [c1], c2):
                        new_system = remaining + [c1, c2]
                        new_alpha = _independence_number_fast(new_system)
                        if new_alpha < current_alpha:
                            current_alpha = new_alpha
                            triples = new_system
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [list(t) for t in triples]
                            improved = True
                            break
            if not improved:
                # Try removing a triple to see if alpha improves
                if len(triples) > 10:
                    i = random.randint(0, len(triples) - 1)
                    new_system = triples[:i] + triples[i+1:]
                    new_alpha = _independence_number_fast(new_system)
                    if new_alpha <= current_alpha:
                        current_alpha = new_alpha
                        triples = new_system
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [list(t) for t in triples]

    return best_system, best_alpha

def construct():
    deadline = time.time() + 115
    random.seed(42)
    
    best_system = None
    best_alpha = 16
    
    # Phase 1: Algebraic design
    alg_system = _algebraic_design()
    if len(alg_system) > 0:
        alpha = _independence_number_fast(alg_system)
        best_alpha = alpha
        best_system = [list(t) for t in alg_system]
    
    # Phase 2: Multiple greedy constructions
    for iteration in range(30):
        if time.time() > deadline:
            break
        system = _greedy_build()
        if len(system) < 5:
            continue
        alpha = _independence_number_fast(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in system]
    
    # Phase 3: Local search on best found
    if best_system is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            improved_system, improved_alpha = _local_search(
                best_system, time_limit=min(remaining_time - 2, 40)
            )
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_system = improved_system
    
    # Phase 4: More iterations if time remains
    while time.time() < deadline - 5:
        system = _greedy_build()
        if len(system) < 5:
            continue
        alpha = _independence_number_fast(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = [list(t) for t in system]
            if time.time() < deadline - 10:
                improved_system, improved_alpha = _local_search(
                    best_system, time_limit=min(deadline - time.time() - 10, 15)
                )
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_system = improved_system
    
    if best_system is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_system

# EVOLVE-BLOCK-END
