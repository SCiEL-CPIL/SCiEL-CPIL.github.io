"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Return exactly 12 distinct integers in [0,128] minimizing the largest
    Sidon subset size (where all unordered pair sums including diagonals are unique)."""
    
    deadline = time.time() + 75  # 75s safety margin before 90s kill
    
    def is_sidon(subset):
        """Check if a subset has all unordered pair sums unique (including diagonals).
        For a set S, we need all sums s_i + s_j for i <= j to be distinct."""
        n = len(subset)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def max_sidon_size(ambient):
        """Find the largest subset of ambient with all unordered pair sums unique."""
        n = len(ambient)
        # Try subsets in decreasing size order
        for size in range(n, 0, -1):
            for subset in combinations(ambient, size):
                if is_sidon(list(subset)):
                    return size
        return 0
    
    def count_pair_sum_collisions(ambient):
        """Count how many times a pair sum is repeated among all unordered pairs (i<=j)."""
        n = len(ambient)
        sums = {}
        for i in range(n):
            for j in range(i, n):
                s = ambient[i] + ambient[j]
                sums[s] = sums.get(s, 0) + 1
        # A collision is when count > 1; total excess collisions
        collisions = sum(count - 1 for count in sums.values() if count > 1)
        return collisions
    
    def evaluate(ambient):
        """Evaluate the ambient set: return (max_sidon_size, collision_count)."""
        ms = max_sidon_size(ambient)
        col = count_pair_sum_collisions(ambient)
        return ms, col
    
    # Strategy: Start with a base set that has many additive relations,
    # then use local search to maximize the minimum Sidon subset size.
    # We want to MINIMIZE the max Sidon subset size, so we want sets where
    # even small subsets tend to have collisions.
    
    # Key insight: If we can make many 3-element subsets have a collision
    # (i.e., a+b = c+d within the 3 elements, which for 3 elements means
    # a+a = b+c or similar), then no 3-element subset is Sidon, so max size <= 2.
    
    # For 3 elements {a, b, c} sorted: pairs (i<=j) are (a,a),(a,b),(a,c),(b,b),(b,c),(c,c)
    # Sums: 2a, a+b, a+c, 2b, b+c, 2c
    # Collision if any two are equal:
    # 2a = a+b => a=b (impossible, distinct)
    # 2a = a+c => a=c (impossible)
    # 2a = 2b => a=b (impossible)
    # 2a = b+c => possible! a = (b+c)/2, i.e., a is midpoint
    # a+b = a+c => b=c (impossible)
    # a+b = 2b => a=b (impossible)
    # a+b = b+c => a=c (impossible)
    # a+b = 2c => possible! a+b = 2c
    # a+c = 2b => possible! a+c = 2b, i.e., b is midpoint of a,c
    # a+c = b+c => a=b (impossible)
    # a+c = 2c => a=c (impossible)
    # 2b = b+c => b=c (impossible)
    # 2b = 2c => b=c (impossible)
    # b+c = 2c => b=c (impossible)
    
    # So for 3 distinct sorted elements a<b<c, collision iff:
    # 2a = b+c (a is "too small", actually this means b+c = 2a, but since b>a, c>a, b+c > 2a, so impossible for sorted)
    # Wait, let me recheck. For sorted a<b<c:
    # 2a < a+b < a+c < 2b? Not necessarily.
    # Let's just check: a+c = 2b means b = (a+c)/2, so b is midpoint. This is the arithmetic progression case.
    # a+b = 2c: since a<c and b<c, a+b < 2c always for positive... wait a+b could equal 2c only if a+b=2c, but a<c, b<c so a+b < 2c. So this is impossible for distinct positive with a<b<c.
    # 2a = b+c: since a<b and a<c, 2a < a+b and 2a < a+c, but b+c > 2a since b>a and c>a. So 2a < b+c. Impossible.
    # So for 3 distinct sorted positive integers, the ONLY collision is when they form an arithmetic progression: a+c = 2b.
    
    # Therefore, to prevent any 3-element subset from being Sidon, we need every triple to be an AP.
    # This is only possible if all elements form a single AP. But then 4 elements of an AP: 
    # {a, a+d, a+2d, a+3d}: check if Sidon.
    # Sums: 2a, 2a+d, 2a+2d, 2a+3d, 2a+2d, 2a+3d, 2a+4d, 2a+3d, 2a+4d, 2a+5d, 2a+6d
    # Wait let me list properly for {a, a+d, a+2d, a+3d}:
    # (a,a)=2a, (a,a+d)=2a+d, (a,a+2d)=2a+2d, (a,a+3d)=2a+3d
    # (a+d,a+d)=2a+2d, (a+d,a+2d)=2a+3d, (a+d,a+3d)=2a+4d
    # (a+2d,a+2d)=2a+4d, (a+2d,a+3d)=2a+5d
    # (a+3d,a+3d)=2a+6d
    # Collisions: 2a+2d appears twice, 2a+3d appears twice, 2a+4d appears twice. So not Sidon.
    
    # So a pure AP of length 4 is NOT Sidon. What's the max Sidon subset of an AP of length 12?
    # An AP {a, a+d, ..., a+11d}. Scaling by d, assume d=1: {0,1,...,11} shifted.
    # The max Sidon subset of {0,...,11} (with diagonal sums) - we need to find this.
    
    # Let's think about what sets minimize the max Sidon subset size.
    # Lower bound: Can we get max Sidon size = 2? That means no 3-element subset is Sidon.
    # As shown, a 3-element set is non-Sidon iff it's an AP. So we need every triple to be an AP.
    # This forces the entire set to be an AP. For an AP of 12 elements, is there a 3-element non-AP subset?
    # No! Every 3-element subset of an AP is itself an AP (in the sense that the middle element is the average).
    # Wait, that's not right. {0, 1, 3} from AP {0,1,2,3,...} - is this an AP? 0+3=3, 2*1=2, 3≠2. So NO, {0,1,3} is NOT an AP.
    # So {0,1,3} would be Sidon! Let me verify: sums are 0,1,3,2,4,6 -> {0,1,2,3,4,6} all distinct. Yes, Sidon.
    # So an AP of 12 elements DOES have 3-element Sidon subsets.
    
    # So max Sidon size >= 3 for any 12-element set? Let me think...
    # Actually, we need to be more careful. The question is: what's the minimum possible max Sidon subset size over all 12-element subsets of [0,128]?
    
    # Let me try a different approach: use randomized search with local improvement.
    
    # Start with various initial sets and try to minimize max_sidon_size.
    
    best_set = None
    best_ms = 13  # start high
    best_col = 0
    
    # Initial candidates
    candidates = []
    
    # Candidate 1: Dense cluster (many close values -> many sum collisions)
    candidates.append(list(range(12)))  # 0-11
    
    # Candidate 2: Two clusters
    candidates.append([0,1,2,3,4,5,60,61,62,63,64,65])
    
    # Candidate 3: AP with small difference
    candidates.append(list(range(0, 12*2, 2)))  # 0,2,4,...,22
    
    # Candidate 4: Mix of dense and sparse
    candidates.append([0,1,2,3,10,11,20,21,40,41,80,81])
    
    # Candidate 5: All even numbers in a range
    candidates.append([2*i for i in range(12)])
    
    # Candidate 6: Quadratic residues mod something
    candidates.append([i*i % 129 for i in range(12)])
    
    # Remove duplicates and ensure validity
    seen = set()
    valid_candidates = []
    for cand in candidates:
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            valid_candidates.append(cand)
    
    # Evaluate initial candidates
    for cand in valid_candidates:
        ms, col = evaluate(cand)
        if ms < best_ms or (ms == best_ms and col > best_col):
            best_ms = ms
            best_col = col
            best_set = list(cand)
    
    # Local search: try to improve by swapping elements
    random.seed(42)
    current = list(best_set)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations % 1000 == 0:
            print(f"Iter {iterations}, best_ms={best_ms}")
        
        # Pick a random element to change
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        
        if new_val == old_val:
            continue
        
        # Check if new_val is already in the set
        if new_val in current:
            continue
        
        current[idx] = new_val
        
        ms, col = evaluate(current)
        
        # Accept if better (lower ms, or same ms with higher col)
        if ms < best_ms or (ms == best_ms and col > best_col):
            best_ms = ms
            best_col = col
            best_set = list(current)
        else:
            # Revert
            current[idx] = old_val
        
        # Occasionally do a random restart from best
        if iterations % 500 == 0:
            current = list(best_set)
            # Perturb
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in current:
                    current[idx] = new_val
    
    # Final polish: try all single-element changes from best_set
    if time.time() < deadline - 5:
        for idx in range(12):
            old_val = best_set[idx]
            for new_val in range(129):
                if new_val == old_val or new_val in best_set:
                    continue
                test_set = list(best_set)
                test_set[idx] = new_val
                ms, col = evaluate(test_set)
                if ms < best_ms or (ms == best_ms and col > best_col):
                    best_ms = ms
                    best_col = col
                    best_set = test_set
                    break
    
    # Ensure we return exactly 12 distinct integers in [0,128]
    result = sorted(best_set)
    assert len(result) == 12
    assert len(set(result)) == 12
    assert all(0 <= x <= 128 for x in result)
    
    return result

# EVOLVE-BLOCK-END
