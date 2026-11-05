"""
Branch-and-bound maximum independent set on a conflict graph.
Nodes: integers 2..128.
Edge (i,j) exists if placing both i and j in the sequence (in sorted order)
would cause any consecutive block product to be duplicated.
We search for the largest subset with no internal conflicts.
"""
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 128  # max value
    nodes = list(range(2, N + 1))  # 2..128
    n_nodes = len(nodes)  # 127

    # Precompute prime factorization for each node (useful for product checks)
    # Actually, we'll work directly with products.

    # Key insight: for a sorted sequence S = [s1, s2, ..., sk], the consecutive
    # block products are all s_i * s_{i+1} * ... * s_j for 1 <= i <= j <= k.
    # Two blocks conflict if their products are equal.
    #
    # For branch-and-bound, we need to check: given a partial assignment (a set
    # of chosen numbers, sorted), can we add a new number x?
    # This is expensive to check dynamically. Instead, we precompute pairwise
    # and higher-order conflicts.
    #
    # Actually, the conflict is NOT purely pairwise. Three numbers a < b < c
    # might all be pairwise compatible but together create a duplicate product
    # (e.g., a*b == c*d for some d). So we need a different approach.
    #
    # Better approach: Use a smarter greedy + branch-and-bound hybrid.
    # 
    # Alternative structural approach: 
    # The constraint is that all O(n^2) block products are distinct.
    # For a sequence of length L, there are L*(L+1)/2 products.
    # The products can be very large (up to 128^L), so we can use Python big ints.
    #
    # Let's use a different framing: 
    # Build the sequence left-to-right. At each step, we choose the next element.
    # The state is (current sequence, set of all block products so far).
    # This is exponential in the number of products, but L is at most ~15-20.
    #
    # Actually, let's think about what lengths are achievable.
    # For L elements, we need L*(L+1)/2 distinct products.
    # The smallest possible products are the singletons: 2,3,...,L+1 (if we use small numbers).
    # But products grow fast. The question is how many elements we can fit.
    #
    # Let's use a different strategy entirely: 
    # Use a SAT/ILP-like approach with backtracking, but with strong pruning.
    # 
    # Actually, the most promising structural change: 
    # Use a "product set" representation and do depth-first search with memoization
    # on the set of used products. But the product set is too large for memoization.
    #
    # Let me try: greedy with lookahead + simulated annealing on the product set.
    # 
    # Actually, let's go with the conflict graph idea but make it work by
    # precomputing for each pair of subsets... no, that's too big.
    #
    # NEW PLAN: 
    # The key observation is that if we have a sorted sequence, the block products
    # are determined. Let's use a branch-and-bound where:
    # - We decide which numbers to include (subset of 2..128)
    # - The sequence is the sorted version of the subset
    # - We check validity by computing all block products
    # - We use an upper bound: |current subset| + |remaining candidates|
    # - We order candidates by a heuristic (e.g., prefer numbers with fewer conflicts)
    #
    # To make this fast, we need a fast validity check and good pruning.
    # 
    # Let's precompute: for each number x in 2..128, its "conflict signature"
    # with respect to products. Actually, let's just do DFS with the following:
    # - State: (index in sorted candidate list, current chosen set, current product set)
    # - But product set is expensive to copy.
    #
    # Let me try a completely different angle:
    # Use the fact that for strictly increasing sequences, the block products
    # are all distinct IF AND ONLY IF no two subarrays have the same product.
    # This is equivalent to saying the sequence is "product-unique" for all subarrays.
    #
    # Known result: For sequences of prime numbers, all block products are distinct
    # (by unique factorization). So we can always use all primes in [2,128].
    # There are 30 primes in [2,128]. So we can always get at least 30.
    #
    # Can we do better than 30? Yes, by including composite numbers that don't
    # create duplicate products.
    #
    # Let's use a branch-and-bound that starts with all primes and tries to add
    # composites, checking validity incrementally.
    #
    # Actually, the cleanest structural change: 
    # Use a "conflict matrix" approach where we precompute for every pair of 
    # numbers whether they can coexist, and for every triple whether they can
    # coexist, etc. But this is intractable for higher orders.
    #
    # Let me just implement a well-optimized DFS with strong pruning:
    # - Sort candidates by a heuristic (primes first, then by number of prime factors)
    # - At each step, try to add the next candidate
    # - Maintain the current sequence and the set of all block products
    # - Prune if the maximum possible size (current + remaining) <= best found
    # - Use time limit

    def get_block_products(seq):
        """Return set of all consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid_seq(seq):
        """Check strictly increasing, in range, and all block products distinct."""
        if not seq:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        products = get_block_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    # Prime sieve
    def is_prime(n):
        if n < 2:
            return False
        for d in range(2, int(n**0.5) + 1):
            if n % d == 0:
                return False
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]

    # Heuristic ordering: primes first (they're "safe"), then composites by 
    # number of distinct prime factors (fewer = safer)
    def count_distinct_pf(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Order: primes first (ascending), then composites by (num distinct pf, value)
    ordered_candidates = sorted(primes) + sorted(composites, key=lambda x: (count_distinct_pf(x), x))

    best_seq = []
    best_len = 0

    # Start with greedy: add each candidate if it doesn't create conflicts
    greedy_seq = []
    greedy_products = set()
    for x in ordered_candidates:
        # Compute new block products if we add x to the end
        # New products: x, x*last, x*last*second_last, ...
        new_prods = [x]
        prod = x
        for a in reversed(greedy_seq):
            prod *= a
            new_prods.append(prod)
        if not any(p in greedy_products for p in new_prods):
            greedy_seq.append(x)
            greedy_products.update(new_prods)

    if is_valid_seq(greedy_seq):
        best_seq = greedy_seq[:]
        best_len = len(greedy_seq)

    # Now do branch-and-bound to try to find a longer sequence
    # We'll use DFS over the ordered candidates, deciding include/exclude
    # with pruning based on upper bound

    # For efficiency, we'll work with the greedy result as a starting point
    # and try to improve it by exploring alternatives

    # Actually, let's do a full DFS with good pruning
    # State: (idx, current_seq, current_products)
    # But current_products is a set of big ints, expensive to copy

    # Alternative: use a recursive function that modifies in place with backtracking

    # Let's use a different approach: iterative deepening on the target length
    # Start from best_len+1 and try to find a sequence of that length

    # For each target length L, we need to find a subset of L numbers from 2..128
    # such that all block products are distinct.
    # 
    # Use DFS with the following pruning:
    # 1. If current size + remaining candidates < target: prune
    # 2. If time exceeds deadline: stop
    # 3. Use a heuristic to order candidates at each step

    # This is still potentially exponential. Let's add more structure.
    #
    # Key insight for pruning: if we've chosen a set S, the block products are
    # determined. When we add a new element x, the new block products are
    # x * (all suffix products of S). If any of these are already in the product
    # set, we can't add x.
    #
    # Let's implement this with careful backtracking.

    import sys
    sys.setrecursionlimit(10000)

    deadline_reached = [False]

    def dfs(idx, cur_seq, cur_products, target_len):
        """
        DFS to find a sequence of length target_len.
        idx: index into ordered_candidates
        cur_seq: current sequence (sorted)
        cur_products: set of all block products for cur_seq
        Returns True if found, False otherwise.
        """
        if deadline_reached[0]:
            return False

        if len(cur_seq) == target_len:
            # Check validity
            if is_valid_seq(cur_seq):
                return True
            return False

        remaining = target_len - len(cur_seq)
        if idx + remaining > n_nodes:
            return False

        if time.time() - start_time > DEADLINE:
            deadline_reached[0] = True
            return False

        # Try to include ordered_candidates[idx]
        x = ordered_candidates[idx]

        # Check if x can be added to cur_seq
        # New block products: x, x*cur_seq[-1], x*cur_seq[-1]*cur_seq[-2], ...
        new_prods = []
        prod = x
        for a in reversed(cur_seq):
            prod *= a
            new_prods.append(prod)
        new_prods.append(x)  # singleton

        # Check for conflicts
        conflict = False
        for p in new_prods:
            if p in cur_products:
                conflict = True
                break

        if not conflict:
            # Also check that x > cur_seq[-1] (strictly increasing)
            if not cur_seq or x > cur_seq[-1]:
                new_seq = cur_seq + [x]
                new_products = cur_products | set(new_prods)
                if dfs(idx + 1, new_seq, new_products, target_len):
                    return True
                # Backtrack: no need to explicitly undo since we're using new objects

        # Try to exclude ordered_candidates[idx]
        if dfs(idx + 1, cur_seq, cur_products, target_len):
            return True

        return False

    # Try to find sequences longer than best_len
    for target_len in range(best_len + 1, 50):
        if time.time() - start_time > DEADLINE:
            break
        if deadline_reached[0]:
            break

        # Upper bound check: can we even fit target_len numbers?
        if target_len > n_nodes:
            break

        found = dfs(0, [], set(), target_len)
        if found:
            # We need to retrieve the actual sequence
            # Let's modify dfs to store the result
            pass

    # The above DFS doesn't store the result. Let me redo this properly.

    best_found = best_seq[:]
    best_found_len = best_len

    def dfs_store(idx, cur_seq, cur_products, target_len):
        """DFS that stores the result in best_found."""
        nonlocal best_found, best_found_len

        if deadline_reached[0]:
            return False

        if len(cur_seq) == target_len:
            if is_valid_seq(cur_seq):
                best_found = cur_seq[:]
                best_found_len = target_len
                return True
            return False

        remaining = target_len - len(cur_seq)
        if idx + remaining > n_nodes:
            return False

        if time.time() - start_time > DEADLINE:
            deadline_reached[0] = True
            return False

        # Try to include ordered_candidates[idx]
        x = ordered_candidates[idx]

        new_prods = []
        prod = x
        for a in reversed(cur_seq):
            prod *= a
            new_prods.append(prod)
        new_prods.append(x)

        conflict = any(p in cur_products for p in new_prods)

        if not conflict and (not cur_seq or x > cur_seq[-1]):
            new_seq = cur_seq + [x]
            new_products = cur_products | set(new_prods)
            if dfs_store(idx + 1, new_seq, new_products, target_len):
                return True

        # Try to exclude
        if dfs_store(idx + 1, cur_seq, cur_products, target_len):
            return True

        return False

    for target_len in range(best_len + 1, 40):
        if time.time() - start_time > DEADLINE:
            break
        if deadline_reached[0]:
            break

        if target_len > n_nodes:
            break

        # Quick upper bound: even with all remaining, can we reach target_len?
        # This is trivially true since we have 127 candidates.
        # Better bound: the number of candidates that could possibly be added
        # given the current product set is complex. Skip for now.

        found = dfs_store(0, [], set(), target_len)
        if found:
            best_seq = best_found[:]
            best_len = best_found_len
            # Try to go even longer
            continue
        else:
            break

    # Final validation
    if not is_valid_seq(best_seq):
        # Fall back to greedy
        best_seq = greedy_seq[:]

    return best_seq

