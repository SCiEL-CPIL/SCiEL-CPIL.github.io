"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s kill

    def check_time():
        return time.time() > deadline

    def score_set(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0

        n = len(U_set)
        if n > 2000:
            return 0.0  # Too slow to compute

        sumset = set()
        diffset = set()

        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)

        if not sumset:
            return 0.0

        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    def generate_geometric_set(b, m):
        """Generate U = { floor(b^k) | k=0...m }"""
        U = set()
        for k in range(m + 1):
            val = int(math.floor(b ** k))
            U.add(val)
        return sorted(U)

    def generate_sporadic_set():
        """
        Generate a set based on a sporadic construction that tends to have
        large difference sets relative to sumsets.
        One known family: use numbers with sparse binary representations
        where the set is constructed to minimize sum collisions.
        
        Try: U = { floor(c * phi^k) } where phi = golden ratio, c is a scale.
        Or: use base-3 digits with only 0 and 1 (Cantor-like).
        """
        # Cantor-like: numbers with base-3 digits 0 or 1
        # For k digits, we get 2^k numbers in [0, 3^k - 1]
        # Let's try a few values of k
        best_score = 0.0
        best_set = None
        
        for k in range(1, 20):
            if check_time():
                break
            
            # Generate all numbers with k base-3 digits, each 0 or 1
            U = set()
            for mask in range(1 << k):
                val = 0
                for i in range(k):
                    if mask & (1 << i):
                        val += 3 ** i
                U.add(val)
            
            U_list = sorted(U)
            if 0 not in U_list:
                U_list.insert(0, 0)
            
            s = score_set(U_list)
            if s > best_score:
                best_score = s
                best_set = U_list
        
        return best_score, best_set

    def generate_sporadic_set_v2():
        """
        Another sporadic construction: use a geometric progression with an irrational ratio
        and round to integers. This avoids periodic structure that causes sum collisions.
        """
        best_score = 0.0
        best_set = None
        
        # Try several irrational ratios
        ratios = [
            math.sqrt(2),
            math.sqrt(3),
            math.pi / 2,
            (1 + math.sqrt(5)) / 2,  # golden ratio
            math.e / 2,
        ]
        
        for ratio in ratios:
            if check_time():
                break
            
            for m in range(5, 50):
                if check_time():
                    break
                
                U = set()
                for k in range(m + 1):
                    val = int(math.floor(ratio ** k))
                    U.add(val)
                
                U_list = sorted(U)
                s = score_set(U_list)
                if s > best_score:
                    best_score = s
                    best_set = U_list
        
        return best_score, best_set

    best_score = 0.0
    best_set = None

    # Strategy 1: Search over geometric progressions U = { floor(b^k) }
    # Discretize m, ternary search over b
    for m in range(3, 60):
        if check_time():
            break
        
        # Coarse grid search over b in log space
        log_lo = math.log(1.02)
        log_hi = math.log(30.0)
        
        best_b_local = None
        best_score_local = -1
        
        n_grid = 15
        for i in range(n_grid):
            log_b = log_lo + (log_hi - log_lo) * i / (n_grid - 1)
            b = math.exp(log_b)
            U = generate_geometric_set(b, m)
            s = score_set(U)
            if s > best_score_local:
                best_score_local = s
                best_b_local = b
        
        if best_b_local is None:
            continue
        
        # Refine with ternary search
        lo_ref = max(1.02, best_b_local * 0.85)
        hi_ref = min(30.0, best_b_local * 1.15)
        
        for _ in range(35):
            if check_time():
                break
            m1 = lo_ref + (hi_ref - lo_ref) / 3.0
            m2 = hi_ref - (hi_ref - lo_ref) / 3.0
            U1 = generate_geometric_set(m1, m)
            U2 = generate_geometric_set(m2, m)
            s1 = score_set(U1)
            s2 = score_set(U2)
            if s1 > s2:
                hi_ref = m2
            else:
                lo_ref = m1
        
        b_final = (lo_ref + hi_ref) / 2.0
        U_final = generate_geometric_set(b_final, m)
        s_final = score_set(U_final)
        
        if s_final > best_score:
            best_score = s_final
            best_set = U_final

    # Strategy 2: Sporadic constructions
    if not check_time():
        s2, set2 = generate_sporadic_set()
        if s2 > best_score:
            best_score = s2
            best_set = set2

    if not check_time():
        s3, set3 = generate_sporadic_set_v2()
        if s3 > best_score:
            best_score = s3
            best_set = set3

    # Fallback
    if best_set is None:
        return [0, 1, 2]

    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.insert(0, 0)

    return sorted(best_set)

# EVOLVE-BLOCK-END
