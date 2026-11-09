import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s kill

    # Fallback: known valid set of 12 elements (all prime powers, no collisions)
    fallback_set = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Wait, let me verify the fallback is valid.
    # Elements: 2^1, 2^2, 2^4, 3^1, 3^2, 5^1, 5^2, 7^1, 11^1, 13^1, 17^1, 19^1, 23^1, 29^1, 31^1, 37^1
    # That's 16 elements. Are all subset products distinct?
    # Yes, because each element is a power of a distinct prime (except 2,4,16 share prime 2; 3,9 share 3; 5,25 share 5).
    # For prime 2: exponents available are {1,2,4}. Subset sums of {1,2,4} are all distinct (0,1,2,3,4,5,6,7).
    # For prime 3: exponents {1,2}. Subset sums: 0,1,2,3. All distinct.
    # For prime 5: exponents {1,2}. Subset sums: 0,1,2,3. All distinct.
    # Other primes: exponent {1}. Subset sums: 0,1. Distinct.
    # Since the primes are independent, the combined product is unique. So 16 is a valid fallback.
    # Let me double check the count: [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37] -> len is 16.
    # The current program's fallback was 12. This is better.

    # Primes in 2..40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    p_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Precompute exponent vectors for all integers 2..40
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return vec

    # Map: exponent vector tuple -> list of integers that have this vector
    vec_to_nums = {}
    for n in range(2, 41):
        v = tuple(factorize(n))
        if v not in vec_to_nums:
            vec_to_nums[v] = []
        vec_to_nums[v].append(n)

    # Candidate exponent vectors (unique ones)
    unique_vecs = list(vec_to_nums.keys())
    # Sort vectors by "cost" (sum of exponents) to prefer smaller/more packable ones first
    unique_vecs.sort(key=lambda v: (sum(v), v))

    # We want to find the maximum number of vectors we can pick such that
    # all 2^k subset sums are distinct.
    # This is equivalent to: the set of vectors forms a "Sidon-like" set for subset sums.
    # Actually, it's stronger: all subset sums must be distinct. This means the vectors
    # must be linearly independent in the sense of subset-sum uniqueness.
    # A sufficient condition: if we sort vectors and each new vector's minimum coordinate
    # (in some sense) is larger than the sum of previous... but that's too restrictive.
    # Better: use the fact that if we represent each vector as a number in a mixed-radix system
    # where each prime dimension has enough "digits", distinct subset sums correspond to
    # the vectors being "superincreasing" in some embedding.
    # 
    # Actually, the cleanest check: maintain the set of all subset sums (as tuples).
    # For k vectors, there are 2^k subset sums. We need them all distinct.
    # 
    # Strategy: Branch and bound.
    # State: (index in unique_vecs, current list of chosen vectors, current set of subset sums)
    # But the set of subset sums can be large. For k=15, 2^15 = 32768 tuples of length 12.
    # That's manageable in memory but slow to compute.
    #
    # Optimization: Instead of storing full tuples, we can store the subset sums as integers
    # by encoding each vector into a single integer using a base that is large enough.
    # For prime 2, max exponent in 2..40 is 5 (32=2^5). Max subset sum of exponents for prime 2:
    # if we pick all 2-powers: 2^1,2^2,2^3,2^4,2^5 -> exponents 1,2,3,4,5 -> sum=15.
    # For prime 3: max exponent is 3 (27=3^3). Sum if we pick 3,9,27: 1+2+3=6.
    # For prime 5: max exponent 2 (25). Sum: 1+2=3.
    # For primes >=7: max exponent 1. Sum: at most 1 per prime.
    # So we can encode each subset sum as an integer:
    # code = sum(exp[i] * base[i]) for i in range(num_primes)
    # where base[i] is chosen so that no overflow occurs.
    # base[0] (for prime 2) must be > max_sum_exp[0] = 15. Use 16.
    # base[1] (for prime 3) must be > max_sum_exp[1] = 6. Use 8.
    # base[2] (for prime 5) must be > max_sum_exp[2] = 3. Use 4.
    # base[i] for i>=3: max_sum_exp[i] <= number_of_chosen_vectors_with_prime_i.
    # Since each vector has at most exponent 1 for primes >=7, and we can have at most 12 such primes,
    # max sum is 12. Use base 13 for each.
    # Total bits: log2(16*8*4*13^9) = 4+3+2+9*3.7 = 43 bits. Fits in Python int easily.

    # Compute max possible exponent sum per prime across ALL candidates 2..40
    max_exp_sum = [0] * num_primes
    for v in unique_vecs:
        for i in range(num_primes):
            max_exp_sum[i] += v[i]
    # But we won't pick ALL vectors. A tighter bound: max_exp_sum[i] is the sum if we picked all.
    # For encoding, we just need a base that prevents collision. Use max_exp_sum[i] + 1 as base.
    bases = [max_exp_sum[i] + 1 for i in range(num_primes)]

    def encode_vector(vec):
        code = 0
        for i in range(num_primes):
            code = code * bases[i] + vec[i]
        return code

    # Precompute encoded form of each unique vector
    vec_codes = [(v, encode_vector(v)) for v in unique_vecs]

    # Branch and bound
    best_count = 0
    best_vecs = []

    # Upper bound: for a given state, the maximum additional vectors we can add is
    # len(remaining_vecs). So if current_count + len(remaining) <= best_count, prune.

    # To make the search faster, we can use a greedy lower bound first.
    # Greedy: pick vectors in order, check compatibility.
    def is_compatible(subset_sums_set, new_vec_code):
        # Check if new_vec_code is in subset_sums_set
        if new_vec_code in subset_sums_set:
            return False
        # Check if s + new_vec_code is in subset_sums_set for any s in subset_sums_set
        # But we need to be careful: we're adding a new vector, so new subset sums are
        # {s + new_vec_code for s in subset_sums_set}. We need none of these to be in
        # subset_sums_set, AND they must be distinct from each other (they are, since s are distinct).
        for s in subset_sums_set:
            if s + new_vec_code in subset_sums_set:
                return False
        return True

    def add_to_sums(subset_sums_set, new_vec_code):
        new_sums = set(subset_sums_set)
        for s in subset_sums_set:
            new_sums.add(s + new_vec_code)
        return new_sums

    # Greedy to get a good initial solution
    greedy_sums = {0}
    greedy_vecs = []
    for v, code in vec_codes:
        if is_compatible(greedy_sums, code):
            greedy_vecs.append(v)
            greedy_sums = add_to_sums(greedy_sums, code)
    best_count = len(greedy_vecs)
    best_vecs = list(greedy_vecs)

    # Now do branch and bound to try to improve
    # We'll do a DFS with pruning.
    # To avoid explosion, we limit the search depth or use time.

    def dfs(idx, current_vecs, current_sums):
        nonlocal best_count, best_vecs, deadline

        if time.time() > deadline:
            return

        if len(current_vecs) > best_count:
            best_count = len(current_vecs)
            best_vecs = list(current_vecs)

        # Prune
        if idx >= len(vec_codes):
            return
        if len(current_vecs) + (len(vec_codes) - idx) <= best_count:
            return

        # Try to include vec_codes[idx]
        v, code = vec_codes[idx]
        if is_compatible(current_sums, code):
            new_sums = add_to_sums(current_sums, code)
            current_vecs.append(v)
            dfs(idx + 1, current_vecs, new_sums)
            current_vecs.pop()

        # Try to exclude vec_codes[idx]
        dfs(idx + 1, current_vecs, current_sums)

    # Start DFS from the greedy solution? No, DFS from scratch might be better for finding optimal.
    # But DFS from scratch could be slow. Let's try DFS but with a time limit.
    # Actually, the greedy solution is already a lower bound. Let's try to beat it.

    # To make DFS more effective, we can start from the greedy solution and try to swap/improve.
    # But for simplicity, let's do a full DFS with time limit.
    # The number of unique vectors is at most 39, but many are duplicates in terms of vector.
    # Let's count: unique_vecs length.
    # For now, let's just run DFS.

    # Actually, DFS from scratch with 39 candidates and pruning might be feasible.
    # Let's try.

    dfs(0, [], {0})

    # Map best_vecs back to integers
    # For each vector in best_vecs, pick one integer that has this vector.
    # We need to ensure we pick distinct integers, but since vectors are unique,
    # and each vector maps to at least one integer, we can just pick the first one.
    # But wait: could two different vectors map to the same integer? No, factorization is unique.
    # Could the same vector be in best_vecs twice? No, we're picking from unique_vecs without replacement.
    result = []
    for v in best_vecs:
        nums = vec_to_nums[v]
        if nums:
            result.append(nums[0])

    # Verify the result
    def verify(nums):
        if len(nums) < 2 or len(nums) > 18:
            return False
        if any(not isinstance(x, int) for x in nums):
            return False
        if any(x < 2 or x > 40 for x in nums):
            return False
        if len(set(nums)) != len(nums):
            return False
        # Check all subset products are distinct
        products = set()
        for r in range(len(nums) + 1):
            for comb in combinations(nums, r):
                prod = 1
                for x in comb:
                    prod *= x
                if prod in products:
                    return False
                products.add(prod)
        return True

    if not verify(result):
        result = fallback_set

    return result
