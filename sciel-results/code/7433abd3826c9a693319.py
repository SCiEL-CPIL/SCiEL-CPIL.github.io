"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    # Moduli range
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # For each pair of moduli, precompute the set of residues that conflict
    # conflict_mask[i][j] is a bitset of residues r_i such that there exists r_j with (r_i - r_j) % gcd == 0
    # Actually, let's think differently.
    
    # The problem: select a set of (m, r) pairs with distinct m, pairwise disjoint.
    # Two classes (m1, r1) and (m2, r2) conflict if r1 ≡ r2 (mod gcd(m1, m2)).
    
    # For a fixed set of selected moduli S, we need to assign residues to each m in S
    # such that for all pairs (m1, m2) in S, r1 ≠ r2 (mod gcd(m1, m2)).
    
    # This is a constraint satisfaction problem. The score is sum of 1/m over m in S.
    # So we want to find the subset S of moduli (with |S| as large as possible, biased toward small m)
    # for which a valid residue assignment exists.
    
    # Key insight: If gcd(m1, m2) = 1, then any residues work (no constraint).
    # If gcd(m1, m2) = g > 1, then r1 mod g ≠ r2 mod g.
    
    # Let's use a different approach:
    # 1. Sort moduli by 1/m (descending) - prioritize small moduli
    # 2. Use backtracking with constraint propagation to find the maximum weight independent set
    #    in the conflict graph where nodes are (m, r) pairs.
    
    # Actually, the number of (m, r) pairs is sum(m for m in range(20, 201)) = ~20000.
    # That's too many for exact methods.
    
    # Better approach: 
    # - The constraint only depends on residues modulo gcd(m1, m2).
    # - For coprime moduli, no constraint.
    # - For moduli with gcd > 1, the constraint is on residues mod g.
    
    # Let's use a greedy + local search approach with a key improvement:
    # When adding a new class, choose the residue that maximizes the number of future
    # classes that can still be added (lookahead).
    
    # Precompute for each pair of moduli the gcd
    # Also precompute: for each modulus m, which other moduli share gcd > 1 with it
    
    def can_add_residue(assigned, m, r):
        """Check if residue r for modulus m is compatible with all assigned residues."""
        mi = mod_idx[m]
        for (m2, r2) in assigned:
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if g > 1 and (r - r2) % g == 0:
                return False
        return True
    
    def count_future_conflicts(assigned, m, r):
        """Count how many unassigned moduli would be blocked by this choice."""
        mi = mod_idx[m]
        count = 0
        for m2 in moduli:
            if m2 == m or any(cm == m2 for cm, _ in assigned):
                continue
            mj = mod_idx[m2]
            g = gcd_table[mi][mj]
            if g > 1:
                # How many residues r2 of m2 would conflict with r?
                # r2 ≡ r (mod g) means r2 = r + k*g for some k
                # Number of such r2 in [0, m2) is either m2//g or m2//g + 1
                # Specifically, residues r2 where r2 % g == r % g
                # That's exactly m2 // g residues (or m2//g + 1 if m2 % g != 0... no)
                # Actually, the number of integers in [0, m2) congruent to r mod g is:
                # floor((m2 - 1 - r) / g) + 1 if r < m2, else 0
                # But since r < m and we're counting residues of m2, it's just m2 // g
                # (possibly +1)
                # For our purpose, we just need to know if ALL residues of m2 are blocked
                # That happens if m2 // g == 1, i.e., g == m2 or g == m2/2... 
                # Actually, if g >= m2, then all residues conflict. But g = gcd(m, m2) <= min(m, m2) < m2 (since m2 > m typically in our ordering)
                # So g < m2 always (when m != m2), meaning there's always at least one valid residue for m2.
                # So this metric isn't super useful. Let's use a different one.
                pass
        return count
    
    # Better lookahead: for each candidate (m, r), count how many residues remain available
    # for each unassigned modulus. But this is expensive.
    
    # Let's use a simpler but effective approach:
    # Greedy with smart residue selection + multiple restarts with different orderings
    
    def greedy_with_lookahead(order):
        """Greedy construction with lookahead-based residue selection."""
        assigned = []
        used_moduli = set()
        
        for m in order:
            if m in used_moduli:
                continue
            
            # Find all valid residues for m
            valid_residues = []
            for r in range(m):
                if can_add_residue(assigned, m, r):
                    valid_residues.append(r)
            
            if not valid_residues:
                continue
            
            # Choose the residue that leaves the most flexibility for future moduli
            # Metric: for each future modulus m2, count how many residues of m2 are still available
            # after choosing r. Pick r that maximizes the minimum count (bottleneck).
            best_r = valid_residues[0]
            best_min_count = -1
            best_total_count = -1
            
            mi = mod_idx[m]
            
            # Only consider a subset of future moduli for speed
            future_moduli = [m2 for m2 in order if m2 > m and m2 not in used_moduli]
            if len(future_moduli) > 30:
                future_moduli = future_moduli[:30]
            
            for r in valid_residues:
                min_count = float('inf')
                total_count = 0
                for m2 in future_moduli:
                    mj = mod_idx[m2]
                    g = gcd_table[mi][mj]
                    if g == 1:
                        # No constraint, all m2 residues are available
                        count = m2
                    else:
                        # Count residues r2 in [0, m2) such that (r2 - r) % g != 0
                        # i.e., r2 % g != r % g
                        # Number of r2 with r2 % g == r % g is m2 // g (or m2//g + 1)
                        # Let's compute exactly
                        rem = r % g
                        # Count of integers in [0, m2) with x % g == rem
                        if rem < m2:
                            cnt = (m2 - 1 - rem) // g + 1
                        else:
                            cnt = 0
                        count = m2 - cnt
                    min_count = min(min_count, count)
                    total_count += count
                
                if min_count > best_min_count or (min_count == best_min_count and total_count > best_total_count):
                    best_min_count = min_count
                    best_total_count = total_count
                    best_r = r
            
            assigned.append([m, best_r])
            used_moduli.add(m)
        
        return assigned
    
    def local_search(assigned, max_iter=2000):
        """Local search: try to add more classes or swap residues."""
        best = [list(c) for c in assigned]
        best_score = sum(Fraction(1, m) for m, _ in best)
        used_moduli = set(m for m, _ in best)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new class
            added = False
            for m in moduli:
                if m in used_moduli:
                    continue
                # Find a valid residue
                for r in range(m):
                    if can_add_residue(best, m, r):
                        new_best = best + [[m, r]]
                        new_score = best_score + Fraction(1, m)
                        if new_score > best_score:
                            best = new_best
                            best_score = new_score
                            used_moduli.add(m)
                            added = True
                            improved = True
                            break
                if added:
                    break
            
            if not improved:
                # Try removing a class and re-adding with different residue
                # or removing one to allow two others
                for i in range(len(best)):
                    m_i, r_i = best[i]
                    # Try removing this class
                    trial = [c for j, c in enumerate(best) if j != i]
                    trial_score = best_score - Fraction(1, m_i)
                    
                    # Try adding two new classes
                    trial_used = set(m for m, _ in trial)
                    added_any = False
                    for m in moduli:
                        if m in trial_used:
                            continue
                        for r in range(m):
                            if can_add_residue(trial, m, r):
                                trial2 = trial + [[m, r]]
                                trial_score2 = trial_score + Fraction(1, m)
                                if trial_score2 > best_score:
                                    best = trial2
                                    best_score = trial_score2
                                    used_moduli = set(m for m, _ in best)
                                    improved = True
                                    added_any = True
                                    break
                        if added_any:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best, best_score
    
    # Multi-start greedy with different orderings
    best_classes = None
    best_score = Fraction(0, 1)
    
    # Ordering 1: smallest moduli first
    order1 = sorted(moduli, key=lambda m: m)
    classes = greedy_with_lookahead(order1)
    classes, s = local_search(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Ordering 2: largest moduli first (to see if we can get a different solution)
    order2 = sorted(moduli, key=lambda m: -m)
    classes = greedy_with_lookahead(order2)
    classes, s = local_search(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # Ordering 3: prime moduli first (coprime to most others)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in moduli if is_prime(m)]
    composites = [m for m in moduli if not is_prime(m)]
    order3 = primes + composites
    classes = greedy_with_lookahead(order3)
    classes, s = local_search(classes)
    if s > best_score:
        best_score = s
        best_classes = [list(c) for c in classes]
    
    # More random orderings
    import random
    random.seed(123)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Random ordering with bias toward small moduli
        order = list(moduli)
        random.shuffle(order)
        # Bias: sort by (random factor * m) to create varied orderings
        weights = [random.random() * m for m in order]
        order = [m for _, m in sorted(zip(weights, order))]
        
        classes = greedy_with_lookahead(order)
        classes, s = local_search(classes, max_iter=1000)
        if s > best_score:
            best_score = s
            best_classes = [list(c) for c in classes]
    
    # Final local search
    if best_classes is not None:
        best_classes, best_score = local_search(best_classes, max_iter=3000)
    
    return best_classes if best_classes is not None else []
# EVOLVE-BLOCK-END
