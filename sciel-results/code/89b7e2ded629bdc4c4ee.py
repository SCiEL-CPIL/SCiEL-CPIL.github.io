import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (limit is 90s)
    
    def count_nonzero_square(terms):
        """Count nonzero terms in square. terms = list of (degree, coeff)."""
        if not terms:
            return 0
        max_deg = 2 * max(d for d, c in terms)
        sq = {}
        n = len(terms)
        for i in range(n):
            di, ci = terms[i]
            for j in range(i, n):
                dj, cj = terms[j]
                deg = di + dj
                if i == j:
                    sq[deg] = sq.get(deg, 0) + ci * cj
                else:
                    sq[deg] = sq.get(deg, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    def make_poly(terms):
        """Create coefficient list in ascending degree format."""
        if not terms:
            return [1]
        max_deg = max(d for d, c in terms)
        result = [0] * (max_deg + 1)
        for d, c in terms:
            result[d] = c
        return result
    
    def validate(result):
        if result is None:
            return False
        if len(result) > 41:
            return False
        if result[-1] == 0:
            return False
        if not all(isinstance(c, int) for c in result):
            return False
        if not all(-64 <= c <= 64 for c in result):
            return False
        if sum(1 for c in result if c != 0) != 12:
            return False
        return True
    
    best_score = float('inf')
    best_result = None
    
    def update_best(terms):
        nonlocal best_score, best_result
        score = count_nonzero_square(terms)
        if score < best_score:
            best_score = score
            result = make_poly(terms)
            if validate(result):
                best_result = result
        return score
    
    # ============================================================
    # PHASE 1: Structured search for high-collision degree sets
    # ============================================================
    # Key insight: we want degree sets where many pairs (i,j) map to the same sum.
    # This happens with arithmetic progressions, geometric-like structures, etc.
    
    # Try arithmetic progressions with various coefficients
    for gap in range(1, 8):
        for start in range(0, 5):
            if start + 11 * gap > 40:
                continue
            degrees = [start + i * gap for i in range(12)]
            
            # Try all-ones first
            terms = list(zip(degrees, [1]*12))
            update_best(terms)
            
            # Try small coefficient patterns
            for coeff_pattern in [
                [1]*12,
                [1,1,1,1,1,1,1,1,1,1,1,1],
                [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1],
                [1,1,-1,1,1,-1,1,1,-1,1,1,-1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
                [1,1,2,1,1,2,1,1,2,1,1,2],
                [1,2,3,1,2,3,1,2,3,1,2,3],
            ]:
                if max(abs(c) for c in coeff_pattern) > 64:
                    continue
                terms = list(zip(degrees, coeff_pattern))
                update_best(terms)
    
    # Try geometric-like (exponential) degree sets
    for base in [2, 3]:
        for start in range(0, 4):
            degrees = [start + base**i for i in range(12)]
            if max(degrees) > 40:
                continue
            terms = list(zip(degrees, [1]*12))
            update_best(terms)
    
    # Try symmetric/palindromic degree sets
    for gap in [1, 2, 3, 4]:
        for center in range(0, 20):
            # Symmetric around center
            degrees = sorted(set())
            for i in range(6):
                d1 = center - i * gap
                d2 = center + i * gap
                if 0 <= d1 <= 40:
                    degrees.append(d1)
                if 0 <= d2 <= 40 and d2 != d1:
                    degrees.append(d2)
            if len(degrees) >= 12:
                degrees = degrees[:12]
                terms = list(zip(degrees, [1]*12))
                update_best(terms)
    
    # ============================================================
    # PHASE 2: Constraint-based search
    # Strategy: pick a target number of nonzero square terms k,
    # then try to find degree/coeff assignments that achieve it.
    # ============================================================
    
    # For a 12-term polynomial, the square has at most 12*13/2 = 78 pair-contributions
    # but at most 2*max_degree+1 distinct degree slots.
    # Minimum possible: if P is a monomial, 1 term. But we need 12 nonzero terms in P.
    # Theoretical minimum for 12-term P: need to maximize collisions.
    
    # Let's try to find degree sets where the pair-sum multiset has few distinct values.
    # Then optimize coefficients to cancel as many as possible.
    
    def pair_sum_collisions(degrees):
        """Return dict mapping sum -> list of (i,j) pairs."""
        sums = {}
        n = len(degrees)
        for i in range(n):
            for j in range(i, n):
                s = degrees[i] + degrees[j]
                if s not in sums:
                    sums[s] = []
                sums[s].append((i, j))
        return sums
    
    def max_possible_cancellation(degrees):
        """For a given degree set, estimate max cancellations possible.
        Returns (num_distinct_sums, min_nonzero_estimate)."""
        sums = pair_sum_collisions(degrees)
        num_distinct = len(sums)
        # Each sum slot can potentially be zeroed out if we have enough freedom
        # With 12 free integer coefficients, we can satisfy up to 12 linear constraints
        # (roughly), so min nonzero ~ max(1, num_distinct - 11) but that's optimistic
        return num_distinct
    
    # Search for degree sets with minimal distinct pair sums
    # Use a greedy/structured approach
    
    # Try "clumped" degree sets: group terms into clusters
    for n_clusters in [2, 3, 4]:
        for _ in range(200):
            if time.time() > deadline - 40:
                break
            # Random clumped set
            cluster_centers = sorted(random.sample(range(0, 41), n_clusters))
            degrees = []
            for c in cluster_centers:
                n_in_cluster = random.randint(2, 6)
                for _ in range(n_in_cluster):
                    d = c + random.randint(-2, 2)
                    d = max(0, min(40, d))
                    degrees.append(d)
            # Deduplicate and ensure 12 unique
            degrees = sorted(set(degrees))
            while len(degrees) < 12:
                d = random.randint(0, 40)
                if d not in degrees:
                    degrees.append(d)
                    degrees.sort()
            degrees = degrees[:12]
            
            if len(set(degrees)) != 12:
                continue
                
            distinct_sums = len(pair_sum_collisions(degrees))
            if distinct_sums < 30:  # Interesting: fewer distinct sums
                terms = list(zip(degrees, [1]*12))
                update_best(terms)
    
    # ============================================================
    # PHASE 3: Focused local search with coefficient optimization
    # ============================================================
    
    def optimize_coefficients(degrees, max_iters=5000):
        """Given fixed degrees, optimize coefficients to minimize nonzero square terms."""
        coeffs = [1] * 12
        current_score = count_nonzero_square(list(zip(degrees, coeffs)))
        
        coeff_choices = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 12, -12, 16, -16, 24, -24, 32, -32, 48, -48, 64, -64]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            idx = random.randint(0, 11)
            old_c = coeffs[idx]
            new_c = random.choice(coeff_choices)
            if new_c == old_c:
                continue
            coeffs[idx] = new_c
            terms = list(zip(degrees, coeffs))
            new_score = count_nonzero_square(terms)
            if new_score <= current_score:
                current_score = new_score
            else:
                coeffs[idx] = old_c
        
        return coeffs, current_score
    
    def optimize_degrees_and_coeffs(terms, max_iters=3000):
        """Joint optimization of degrees and coefficients."""
        current_score = count_nonzero_square(terms)
        coeff_choices = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 12, -12, 16, -16, 24, -24, 32, -32, 48, -48, 64, -64]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            if move_type < 0.5:
                # Change coefficient
                idx = random.randint(0, 11)
                old_c = terms[idx][1]
                new_c = random.choice(coeff_choices)
                if new_c == old_c:
                    continue
                terms[idx] = (terms[idx][0], new_c)
                new_score = count_nonzero_square(terms)
                if new_score < current_score:
                    current_score = new_score
                elif new_score > current_score:
                    terms[idx] = (terms[idx][0], old_c)
            elif move_type < 0.85:
                # Move degree
                idx = random.randint(0, 11)
                old_d = terms[idx][0]
                new_d = random.randint(0, 40)
                if new_d == old_d:
                    continue
                if any(t[0] == new_d for i, t in enumerate(terms) if i != idx):
                    continue
                terms[idx] = (new_d, terms[idx][1])
                terms.sort(key=lambda x: x[0])
                new_score = count_nonzero_square(terms)
                if new_score < current_score:
                    current_score = new_score
                elif new_score > current_score:
                    terms[idx] = (old_d, terms[idx][1])
                    terms.sort(key=lambda x: x[0])
            else:
                # Swap two degrees
                i, j = random.sample(range(12), 2)
                di, dj = terms[i][0], terms[j][0]
                if di == dj:
                    continue
                terms[i] = (dj, terms[i][1])
                terms[j] = (di, terms[j][1])
                terms.sort(key=lambda x: x[0])
                new_score = count_nonzero_square(terms)
                if new_score < current_score:
                    current_score = new_score
                else:
                    # Revert
                    # Re-find the terms by degree
                    terms.sort(key=lambda x: x[0])
                    # We need to put them back - this is tricky after sort
                    # Just revert by re-sorting and finding
                    terms[i], terms[j] = terms[j], terms[i]
                    terms.sort(key=lambda x: x[0])
                    current_score = count_nonzero_square(terms)
        
        return terms, current_score
    
    # Run focused searches from good starting points
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        if r == 0 and best_result is not None:
            # Start from best found so far
            terms = [(i, best_result[i]) for i in range(len(best_result)) if best_result[i] != 0]
            if len(terms) == 12:
                terms, score = optimize_degrees_and_coeffs(terms, max_iters=5000)
                update_best(terms)
                continue
        
        # Generate diverse starting points
        strategy = random.random()
        
        if strategy < 0.3:
            # Arithmetic progression with random coefficients
            gap = random.choice([1, 2, 3, 4])
            start = random.randint(0, 40 - 11*gap)
            degrees = [start + i * gap for i in range(12)]
            coeffs = [random.choice([1, -1, 2, -2, 3, -3, 4, -4]) for _ in range(12)]
            terms = list(zip(degrees, coeffs))
        elif strategy < 0.5:
            # Random degrees, optimized coefficients
            degrees = sorted(random.sample(range(0, 41), 12))
            coeffs = [1] * 12
            terms = list(zip(degrees, coeffs))
        elif strategy < 0.7:
            # Clumped degrees
            cluster_centers = sorted(random.sample(range(2, 39), random.randint(2, 4)))
            degrees = []
            for c in cluster_centers:
                for _ in range(random.randint(2, 6)):
                    d = max(0, min(40, c + random.randint(-3, 3)))
                    if d not in degrees:
                        degrees.append(d)
            while len(degrees) < 12:
                d = random.randint(0, 40)
                if d not in degrees:
                    degrees.append(d)
            degrees = sorted(degrees[:12])
            coeffs = [random.choice([1, -1, 2, -2]) for _ in range(12)]
            terms = list(zip(degrees, coeffs))
        else:
            # Fully random
            degrees = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8]) for _ in range(12)]
            terms = list(zip(degrees, coeffs))
        
        # Joint optimization
        terms, score = optimize_degrees_and_coeffs(terms, max_iters=3000)
        update_best(terms)
    
    # ============================================================
    # PHASE 4: Targeted search for specific low scores
    # ============================================================
    # Try to achieve score 13, 14, 15, etc.
    # A 12-term polynomial squared has at least 12 terms (diagonal) + off-diagonal.
    # Minimum is actually higher. Let's see what we can achieve.
    
    if time.time() < deadline - 10:
        # Try more aggressive searches
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random restart with more iterations
            degrees = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 12, -12, 16, -16, 24, -24, 32, -32, 48, -48, 64, -64]) for _ in range(12)]
            terms = list(zip(degrees, coeffs))
            terms, score = optimize_degrees_and_coeffs(terms, max_iters=5000)
            update_best(terms)
    
    # Final validation
    if best_result is None:
        # Fallback: simple 12-term polynomial
        best_result = [0] * 41
        for i in range(12):
            best_result[i] = 1
        return best_result
    
    if not validate(best_result):
        # Fallback
        best_result = [0] * 41
        for i in range(12):
            best_result[i] = 1
    
    return best_result
# EVOLVE-BLOCK-END
