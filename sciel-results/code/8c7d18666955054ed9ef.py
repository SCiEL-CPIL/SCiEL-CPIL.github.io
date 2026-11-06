"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B using constraint-based backtracking over sum values."""
    deadline = time.time() + 80  # 80s budget, leaving 10s safety margin
    
    # Precompute sieve for primality
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Generate candidate sum values: primes and odd prime powers in [2, 256]
    # Focus on ODD sums to eliminate factor 2 constraint
    candidates = []
    for s in range(3, 257, 2):  # Only odd sums
        if is_prime[s]:
            candidates.append(s)
        else:
            # Include odd prime powers p^k where p is odd prime and k >= 2
            n = s
            p = None
            for prime in range(3, int(math.isqrt(s)) + 1, 2):
                if s % prime == 0:
                    if p is not None and p != prime:
                        break
                    p = prime
                    n = s
                    while n % prime == 0:
                        n //= prime
                    if n == 1:
                        candidates.append(s)
                        break
            else:
                # s might be an odd prime (handled) or product of distinct primes
                pass
    
    # Remove duplicates and sort
    candidates = list(set(candidates))
    candidates.sort()
    
    # Build coprime compatibility matrix for candidates
    n_cands = len(candidates)
    coprime = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            coprime[i][j] = coprime[j][i] = (_gcd(candidates[i], candidates[j]) == 1)
    
    # Sort candidates by degree in coprime graph (most connected first)
    degrees = [sum(coprime[i]) for i in range(n_cands)]
    order = sorted(range(n_cands), key=lambda i: -degrees[i])
    
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _eval(best_A, best_B)
    
    # Phase 1: Focused search with parity constraint
    # A all odd, B all even -> all sums odd (eliminates factor 2)
    # Or A all even, B all odd -> all sums odd
    
    # Precompute for each a in [1,128], which b values make a+b an odd prime
    odd_prime_b_for_a = {}
    for a in range(1, 129):
        odd_prime_b_for_a[a] = []
        for b in range(1, 129):
            s = a + b
            if 3 <= s <= 256 and s % 2 == 1 and is_prime[s]:
                odd_prime_b_for_a[a].append(b)
    
    # Phase 1a: A all odd, B all even, try to make sums prime
    odd_a_values = [x for x in range(1, 129) if x % 2 == 1]
    even_b_values = [x for x in range(2, 129) if x % 2 == 0]
    
    # Try structured A sets with odd values
    for a1 in odd_a_values[:40]:
        if time.time() > deadline:
            break
        for a2 in [x for x in odd_a_values if x > a1 and x - a1 < 40]:
            if time.time() > deadline:
                break
            for a3 in [x for x in odd_a_values if x > a2 and x - a2 < 40]:
                if time.time() > deadline:
                    break
                
                A = [a1, a2, a3]
                
                # Find B values (even) that make all three sums odd primes
                common_b = set(odd_prime_b_for_a[a1]) & set(odd_prime_b_for_a[a2]) & set(odd_prime_b_for_a[a3])
                common_b_even = [b for b in common_b if b % 2 == 0]
                
                if len(common_b_even) >= 3:
                    b_list = sorted(common_b_even)
                    if len(b_list) <= 15:
                        b_combos = list(combinations(b_list, 3))
                    else:
                        b_combos = [tuple(rng.sample(b_list, 3)) for _ in range(50)]
                        b_combos = list(set(b_combos))
                    
                    for B_tuple in b_combos:
                        if time.time() > deadline:
                            break
                        B = list(B_tuple)
                        score = _eval(A, B)
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return best_A, best_B
    
    # Phase 1b: A all even, B all odd
    even_a_values = [x for x in range(2, 129) if x % 2 == 0]
    odd_b_values = [x for x in range(1, 129) if x % 2 == 1]
    
    for a1 in even_a_values[:40]:
        if time.time() > deadline:
            break
        for a2 in [x for x in even_a_values if x > a1 and x - a1 < 40]:
            if time.time() > deadline:
                break
            for a3 in [x for x in even_a_values if x > a2 and x - a2 < 40]:
                if time.time() > deadline:
                    break
                
                A = [a1, a2, a3]
                
                # Find B values (odd) that make all three sums odd primes
                common_b = set(odd_prime_b_for_a[a1]) & set(odd_prime_b_for_a[a2]) & set(odd_prime_b_for_a[a3])
                common_b_odd = [b for b in common_b if b % 2 == 1]
                
                if len(common_b_odd) >= 3:
                    b_list = sorted(common_b_odd)
                    if len(b_list) <= 15:
                        b_combos = list(combinations(b_list, 3))
                    else:
                        b_combos = [tuple(rng.sample(b_list, 3)) for _ in range(50)]
                        b_combos = list(set(b_combos))
                    
                    for B_tuple in b_combos:
                        if time.time() > deadline:
                            break
                        B = list(B_tuple)
                        score = _eval(A, B)
                        if score > best_score:
                            best_score = score
                            best_A = A[:]
                            best_B = B[:]
                            if best_score >= 9:
                                return best_A, best_B
    
    # Phase 2: Backtracking over sum values to find 9 pairwise-coprime odd sums
    # that can form a 3x3 grid with parity constraint
    
    def _can_form_grid_parity(sums_set):
        """Check if a set of 9 sums can form a 3x3 grid with parity constraint."""
        if len(sums_set) != 9:
            return None
        
        sums = list(sums_set)
        
        for s1 in sums:
            for s2 in sums:
                if s1 == s2:
                    continue
                db = s2 - s1  # b2 - b1
                if db == 0:
                    continue
                
                min_b1 = max(1, s1 - 128)
                max_b1 = min(128, s1 - 1)
                
                for b1 in range(min_b1, max_b1 + 1):
                    a1 = s1 - b1
                    b2 = b1 + db
                    if not (1 <= b2 <= 128):
                        continue
                    
                    remaining = [s for s in sums if s != s1 and s != s2]
                    
                    for i in range(len(remaining)):
                        for j in range(i + 1, len(remaining)):
                            s_a2b1 = remaining[i]
                            s_a3b1 = remaining[j]
                            a2 = s_a2b1 - b1
                            a3 = s_a3b1 - b1
                            if not (1 <= a2 <= 128 and 1 <= a3 <= 128):
                                continue
                            if a2 == a1 or a3 == a1 or a2 == a3:
                                continue
                            
                            s_a2b2 = a2 + b2
                            s_a3b2 = a3 + b2
                            if s_a2b2 not in sums_set or s_a3b2 not in sums_set:
                                continue
                            if s_a2b2 in (s1, s2, s_a2b1, s_a3b1):
                                continue
                            if s_a3b2 in (s1, s2, s_a2b1, s_a3b1):
                                continue
                            
                            used = {s1, s2, s_a2b1, s_a3b1, s_a2b2, s_a3b2}
                            remaining3 = [s for s in sums_set if s not in used]
                            if len(remaining3) != 3:
                                continue
                            
                            for s_a1b3 in remaining3:
                                b3 = s_a1b3 - a1
                                if not (1 <= b3 <= 128):
                                    continue
                                if b3 in (b1, b2):
                                    continue
                                
                                s_a2b3 = a2 + b3
                                s_a3b3 = a3 + b3
                                if s_a2b3 not in sums_set or s_a3b3 not in sums_set:
                                    continue
                                if {s_a2b3, s_a3b3} != set(remaining3) - {s_a1b3}:
                                    continue
                                
                                all_sums = [s1, s2, s_a2b1, s_a3b1, s_a2b2, s_a3b2, s_a1b3, s_a2b3, s_a3b3]
                                if len(set(all_sums)) != 9:
                                    continue
                                if any(s < 2 or s > 256 for s in all_sums):
                                    continue
                                
                                return ([a1, a2, a3], [b1, b2, b3])
        
        return None
    
    # Backtracking search for clique of size 9
    found_grid = None
    backtrack_limit = 50000
    count = 0
    
    def _backtrack_clique(current, start_idx):
        nonlocal found_grid, count
        if time.time() > deadline or count > backtrack_limit:
            return
        
        if len(current) == 9:
            result = _can_form_grid_parity(set(candidates[i] for i in current))
            if result:
                found_grid = result
                return
            return
        
        for idx in range(start_idx, n_cands):
            if count > backtrack_limit or time.time() > deadline:
                return
            count += 1
            i = order[idx]
            if not all(coprime[i][j] for j in current):
                continue
            current.append(i)
            _backtrack_clique(current, idx + 1)
            if found_grid:
                return
            current.pop()
    
    for start_i in order[:30]:
        if found_grid or time.time() > deadline or count > backtrack_limit:
            break
        _backtrack_clique([start_i], order.index(start_i) + 1)
    
    if found_grid:
        A, B = found_grid
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
    
    # Phase 3: Focused random search with parity constraint
    for _ in range(200000):
        if time.time() > deadline:
            break
        # 50% chance: A odd, B even; 50% chance: A even, B odd
        if rng.random() < 0.5:
            A = list(rng.sample(odd_a_values, 3))
            B = list(rng.sample(even_b_values, 3))
        else:
            A = list(rng.sample(even_a_values, 3))
            B = list(rng.sample(odd_b_values, 3))
        
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return best_A, best_B
    
    # Phase 4: Local search with parity preservation
    for _ in range(200000):
        if time.time() > deadline:
            break
        if rng.random() < 0.5:
            idx = rng.randint(0, 2)
            # Preserve parity of A
            if best_A[0] % 2 == 1:
                new_val = rng.choice(odd_a_values)
            else:
                new_val = rng.choice(even_a_values)
            if new_val not in best_A:
                old = best_A[idx]
                best_A[idx] = new_val
                score = _eval(best_A, best_B)
                if score >= best_score:
                    best_score = score
                    if best_score >= 9:
                        return best_A, best_B
                else:
                    best_A[idx] = old
        else:
            idx = rng.randint(0, 2)
            # Preserve parity of B
            if best_B[0] % 2 == 1:
                new_val = rng.choice(odd_b_values)
            else:
                new_val = rng.choice(even_b_values)
            if new_val not in best_B:
                old = best_B[idx]
                best_B[idx] = new_val
                score = _eval(best_A, best_B)
                if score >= best_score:
                    best_score = score
                    if best_score >= 9:
                        return best_A, best_B
                else:
                    best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
