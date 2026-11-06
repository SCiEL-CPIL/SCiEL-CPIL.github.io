import random
import math
import time
from itertools import combinations

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def evaluate_pair(A, B):
    """Return (cardinality, is_valid) for a pair of 3-element lists."""
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    curr_cardinality = len(distinct_sums)
    if curr_cardinality < 6:
        return curr_cardinality, False
    if is_pairwise_coprime(distinct_sums):
        return curr_cardinality, True
    return curr_cardinality, False

def local_search(A, B, deadline, best_card):
    """Hill-climb from (A, B) by trying all single-element replacements systematically."""
    current_A = list(A)
    current_B = list(B)
    curr_card, curr_valid = evaluate_pair(current_A, current_B)
    
    all_nums = list(range(1, 129))
    no_improve = 0
    
    while time.time() < deadline:
        if curr_valid and curr_card == 9:
            return list(current_A), list(current_B)
        
        improved = False
        
        # Try all possible replacements for each element of A
        for idx in range(3):
            old_val = current_A[idx]
            for new_val in all_nums:
                if new_val == old_val:
                    continue
                test_A = list(current_A)
                test_A[idx] = new_val
                if len(set(test_A)) != 3:
                    continue
                card, valid = evaluate_pair(test_A, current_B)
                
                if valid and card > curr_card:
                    current_A = test_A
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    if curr_card == 9:
                        return list(current_A), list(current_B)
                    break
                elif valid and card == curr_card and curr_valid:
                    if random.random() < 0.05:
                        current_A = test_A
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
        
        # Try all possible replacements for each element of B
        for idx in range(3):
            old_val = current_B[idx]
            for new_val in all_nums:
                if new_val == old_val:
                    continue
                test_B = list(current_B)
                test_B[idx] = new_val
                if len(set(test_B)) != 3:
                    continue
                card, valid = evaluate_pair(current_A, test_B)
                
                if valid and card > curr_card:
                    current_B = test_B
                    curr_card, curr_valid = card, True
                    improved = True
                    no_improve = 0
                    if curr_card == 9:
                        return list(current_A), list(current_B)
                    break
                elif valid and card == curr_card and curr_valid:
                    if random.random() < 0.05:
                        current_B = test_B
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
        
        if no_improve >= 3:
            if time.time() > deadline - 2:
                break
            # Perturb one element randomly
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                current_A[idx] = random.choice(all_nums)
                if len(set(current_A)) != 3:
                    current_A[idx] = random.choice(all_nums)
            else:
                idx = random.randint(0, 2)
                current_B[idx] = random.choice(all_nums)
                if len(set(current_B)) != 3:
                    current_B[idx] = random.choice(all_nums)
            curr_card, curr_valid = evaluate_pair(current_A, current_B)
            no_improve = 0
    
    if curr_valid:
        return list(current_A), list(current_B)
    else:
        return list(A), list(B)

def graph_backtrack_search(deadline, max_sums=9):
    """
    Build a compatibility graph of pairwise-coprime sums, then backtrack to find
    9 sums that can be arranged as a 3x3 grid (A+B structure).
    
    Key idea: pick 9 pairwise-coprime sums s_{ij}, then check if they can be
    decomposed as s_{ij} = a_i + b_j for some a_i, b_j in [1,128].
    """
    # Precompute all pairwise-coprime pairs of sums in [3, 256]
    # (sums of two numbers in [1,128] range from 3 to 256)
    all_sums = list(range(3, 257))
    
    # Build compatibility: which sums are pairwise coprime
    coprime_pairs = {}
    for i in range(len(all_sums)):
        coprime_pairs[all_sums[i]] = []
        for j in range(i+1, len(all_sums)):
            if math.gcd(all_sums[i], all_sums[j]) == 1:
                coprime_pairs[all_sums[i]].append(all_sums[j])
                coprime_pairs[all_sums[j]].append(all_sums[i])
    
    # Precompute: for each sum s, what pairs (a,b) with a+b=s, a,b in [1,128]
    # For a 3x3 grid with sums s00,s01,s02,s10,s11,s12,s20,s21,s22:
    # s_{ij} = a_i + b_j
    # This means: s_{ij} - s_{ik} = b_j - b_k (independent of i)
    # And: s_{ij} - s_{lj} = a_i - a_l (independent of j)
    
    # Strategy: pick 3 rows (a values) and 3 cols (b values), check if all 9 sums are pairwise coprime
    # But that's the original problem. Instead, use the graph structure.
    
    # Better: enumerate candidate sum sets of size 9 that are pairwise coprime,
    # then check if they form a valid 3x3 additive grid.
    
    # For efficiency, focus on sums that are primes or prime powers (more likely to be coprime)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Candidate sums: primes in [3, 256] are ideal (always coprime to each other)
    prime_sums = [s for s in all_sums if is_prime(s)]
    # Also include powers of 2, 3, 5, 7, 11, 13 (coprime to primes not dividing them)
    extra_sums = []
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        for k in range(1, 10):
            v = p ** k
            if 3 <= v <= 256 and v not in prime_sums:
                extra_sums.append(v)
    
    candidates = prime_sums + [s for s in extra_sums if s not in prime_sums]
    candidates = sorted(set(candidates))
    
    # Randomly sample subsets of 9 from candidates, check if they form a valid grid
    # A set {s_{ij}} forms a valid grid iff:
    # There exist a0,a1,a2,b0,b1,b2 such that s_{ij} = a_i + b_j
    # Equivalently: s_{ij} + s_{kl} = s_{il} + s_{kj} for all i,j,k,l
    # (the "additive structure" / Latin square condition)
    
    def check_grid(sums9):
        """Check if 9 sums can be arranged as a 3x3 additive grid."""
        # Try all 9! / (3!3!3!) = 1680 ways to assign to grid positions
        # Actually, fix one element and try permutations of the rest
        # s[0] is fixed at position (0,0)
        # Then we need to assign the remaining 8 to positions (0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
        
        # Use the constraint: s_{ij} = a_i + b_j
        # s_{01} - s_{00} = b_1 - b_0
        # s_{02} - s_{00} = b_2 - b_0
        # s_{10} - s_{00} = a_1 - a_0
        # s_{20} - s_{00} = a_2 - a_0
        # s_{11} should = s_{10} + s_{01} - s_{00}
        # s_{12} should = s_{10} + s_{02} - s_{00}
        # s_{21} should = s_{20} + s_{01} - s_{00}
        # s_{22} should = s_{20} + s_{02} - s_{00}
        
        # So pick 4 "free" sums: s00, s01, s02, s10, s20 (5 free), and 4 are determined
        # Actually: s00 is free, then s01, s02, s10, s20 are free (4 more), and s11,s12,s21,s22 are determined
        # Total: 5 free, 4 determined. We need to check if the determined values are in our set.
        
        remaining = list(sums9)
        
        # Try all choices of s00 from remaining
        for i0 in range(9):
            s00 = remaining[i0]
            rest1 = remaining[:i0] + remaining[i0+1:]
            
            for i1 in range(8):
                s01 = rest1[i1]
                rest2 = rest1[:i1] + rest1[i1+1:]
                
                for i2 in range(7):
                    s02 = rest2[i2]
                    rest3 = rest2[:i2] + rest2[i2+1:]
                    
                    for i3 in range(6):
                        s10 = rest3[i3]
                        rest4 = rest3[:i3] + rest3[i3+1:]
                        
                        for i4 in range(5):
                            s20 = rest4[i4]
                            rest5 = rest4[:i4] + rest4[i4+1:]
                            
                            # Compute determined values
                            s11 = s10 + s01 - s00
                            s12 = s10 + s02 - s00
                            s21 = s20 + s01 - s00
                            s22 = s20 + s02 - s00
                            
                            # Check if all determined values are in rest5
                            needed = [s11, s12, s21, s22]
                            available = list(rest5)
                            
                            all_found = True
                            for val in needed:
                                if val in available:
                                    available.remove(val)
                                else:
                                    all_found = False
                                    break
                            
                            if not all_found:
                                continue
                            
                            # Check that a_i, b_j are in [1, 128]
                            a0 = s00 - s01 + s10  # Not quite right, let me recompute
                            # s00 = a0 + b0
                            # s01 = a0 + b1 => b1 = s01 - a0
                            # s10 = a1 + b0 => a1 = s10 - b0
                            # We need: a0 + b0 = s00, a0 + b1 = s01, a1 + b0 = s10
                            # b1 - b0 = s01 - s00
                            # a1 - a0 = s10 - s00
                            # Let a0 = x, b0 = s00 - x
                            # a1 = x + (s10 - s00)
                            # b1 = s00 - x + (s01 - s00) = s01 - x
                            # a2 = x + (s20 - s00)
                            # b2 = s00 - x + (s02 - s00) = s02 - x
                            
                            # Constraints: all a_i, b_j in [1, 128]
                            # a0 = x in [1,128]
                            # b0 = s00 - x in [1,128] => x in [s00-128, s00-1]
                            # a1 = x + s10 - s00 in [1,128] => x in [1-s10+s00, 128-s10+s00]
                            # b1 = s01 - x in [1,128] => x in [s01-128, s01-1]
                            # a2 = x + s20 - s00 in [1,128] => x in [1-s20+s00, 128-s20+s00]
                            # b2 = s02 - x in [1,128] => x in [s02-128, s02-1]
                            
                            lo = max(1, s00 - 128, 1 - s10 + s00, s01 - 128, 1 - s20 + s00, s02 - 128)
                            hi = min(128, s00 - 1, 128 - s10 + s00, s01 - 1, 128 - s20 + s00, s02 - 1)
                            
                            if lo > hi:
                                continue
                            
                            # Found a valid grid!
                            x = lo
                            a0 = x
                            b0 = s00 - x
                            a1 = x + s10 - s00
                            b1 = s01 - x
                            a2 = x + s20 - s00
                            b2 = s02 - x
                            
                            A = [a0, a1, a2]
                            B = [b0, b1, b2]
                            
                            # Verify
                            if len(set(A)) != 3 or len(set(B)) != 3:
                                continue
                            if any(a < 1 or a > 128 for a in A):
                                continue
                            if any(b < 1 or b > 128 for b in B):
                                continue
                            
                            # Final verification
                            card, valid = evaluate_pair(A, B)
                            if valid and card == 9:
                                return A, B
                            
                            # Try other x values
                            for xx in range(lo + 1, hi + 1):
                                a0 = xx
                                b0 = s00 - xx
                                a1 = xx + s10 - s00
                                b1 = s01 - xx
                                a2 = xx + s20 - s00
                                b2 = s02 - xx
                                
                                A = [a0, a1, a2]
                                B = [b0, b1, b2]
                                
                                if len(set(A)) != 3 or len(set(B)) != 3:
                                    continue
                                if any(a < 1 or a > 128 for a in A):
                                    continue
                                if any(b < 1 or b > 128 for b in B):
                                    continue
                                
                                card, valid = evaluate_pair(A, B)
                                if valid and card == 9:
                                    return A, B
                            
                            return None
        return None
    
    # Sample random subsets of 9 from candidates
    random.shuffle(candidates)
    attempts = 0
    while time.time() < deadline:
        if len(candidates) < 9:
            break
        
        # Take a random subset of 9
        subset = random.sample(candidates, 9)
        
        # Check pairwise coprime
        if not is_pairwise_coprime(subset):
            continue
        
        # Check if it forms a valid grid
        result = check_grid(subset)
        if result is not None:
            return result
        
        attempts += 1
        # Occasionally reshuffle to avoid getting stuck
        if attempts % 1000 == 0:
            random.shuffle(candidates)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # Initialize with better structured seeds based on number theory
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_cardinality = 5
    
    # Try structured initializations with number-theoretic motivation
    structured_starts = [
        ([1, 3, 5], [2, 4, 6]),
        ([1, 5, 7], [2, 4, 8]),
        ([1, 7, 11], [2, 6, 10]),
        ([3, 5, 9], [2, 4, 8]),
        ([1, 3, 7], [4, 8, 12]),
        ([1, 9, 25], [2, 10, 26]),
        ([3, 7, 19], [4, 12, 24]),
        ([1, 5, 13], [6, 14, 22]),
        ([7, 11, 23], [2, 8, 16]),
        ([1, 11, 29], [4, 16, 32]),
        ([1, 2, 4], [3, 5, 7]),
        ([1, 4, 9], [2, 5, 8]),
        ([2, 3, 5], [1, 4, 7]),
        ([1, 6, 15], [2, 7, 16]),
        ([3, 8, 17], [2, 5, 14]),
    ]
    
    for sA, sB in structured_starts:
        card, valid = evaluate_pair(sA, sB)
        if valid and card > best_cardinality:
            best_A, best_B = list(sA), list(sB)
            best_cardinality = card
            if card == 9:
                return [best_A, best_B]
    
    # Phase 1: Graph-based backtracking search (new approach)
    if time.time() < deadline - 10:
        result = graph_backtrack_search(min(deadline - 10, time.time() + 25))
        if result is not None:
            A, B = result
            card, valid = evaluate_pair(A, B)
            if valid and card == 9:
                return [A, B]
            if valid and card > best_cardinality:
                best_A, best_B = A, B
                best_cardinality = card
    
    # Phase 2: Random search with local refinement
    random_iterations = 0
    while time.time() < deadline - 5:
        # Alternate or randomize between parity cases
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        card, valid = evaluate_pair(A, B)
        
        if card == 9 and valid:
            return [A, B]
        
        if valid and card > best_cardinality:
            best_A = list(A)
            best_B = list(B)
            best_cardinality = card
            if card == 9:
                return [best_A, best_B]
        
        random_iterations += 1
        
        if random_iterations % 50 == 0:
            la, lb = local_search(best_A, best_B, min(deadline - 2, time.time() + 3), best_cardinality)
            card, valid = evaluate_pair(la, lb)
            if valid and card > best_cardinality:
                best_A, best_B = la, lb
                best_cardinality = card
                if card == 9:
                    return [best_A, best_B]
        
        if card >= 6 and valid and random.random() < 0.2:
            la, lb = local_search(A, B, min(deadline - 2, time.time() + 2), best_cardinality)
            card, valid = evaluate_pair(la, lb)
            if valid and card > best_cardinality:
                best_A, best_B = la, lb
                best_cardinality = card
                if card == 9:
                    return [best_A, best_B]
    
    # Final polish
    if time.time() < deadline:
        la, lb = local_search(best_A, best_B, deadline, best_cardinality)
        card, valid = evaluate_pair(la, lb)
        if valid and card > best_cardinality:
            best_A, best_B = la, lb
            best_cardinality = card
    
    return [best_A, best_B]
