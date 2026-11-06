"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin from 90s limit

    MAX_N = 512
    MAX_SIZE = 48
    MAX_SUM = 24576  # 48 * 512

    # --- Fast Greedy Construction with Bitset DP ---
    def try_greedy(order):
        """
        Greedily builds a valid set.
        Returns the set of selected integers.
        Uses bitsets for subset-sum DP to check mean conditions efficiently.
        """
        selected = []
        n = 0
        dp_all = [1]  # dp_all[k] is bitmask of sums achievable with exactly k elements
        dp_without = {}  # dp_without[a][k] is bitmask of sums achievable with k elements from selected \ {a}

        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= MAX_SIZE:
                break

            valid = True

            # Check 1: Is x the mean of any subset of selected of size >= 2?
            # sum(T) = k * x for some k >= 2, T subset of selected
            for k in range(2, n + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if k < len(dp_all) and ((dp_all[k] >> target) & 1):
                    valid = False
                    break

            if not valid:
                continue

            # Check 2: For each a in selected, is a the mean of a subset involving x?
            # T = {x} U U, U subset of selected \ {a}
            # sum(T) = |T| * a => x + sum(U) = (1 + |U|) * a
            # sum(U) = (1 + |U|) * a - x
            # Let k = |U| + 1 (size of T). Then k >= 2.
            # sum(U) = k * a - x. |U| = k - 1.
            # We check if k*a - x is achievable with k-1 elements from selected \ {a}.
            for a in selected:
                if not valid:
                    break
                dp_w = dp_without[a]
                for k in range(2, n + 1):
                    target = k * a - x
                    if target < 0:
                        continue
                    if target > (k - 1) * 512:
                        continue
                    if k - 1 < len(dp_w) and ((dp_w[k - 1] >> target) & 1):
                        valid = False
                        break

            if not valid:
                continue

            # Add x to selected
            selected.append(x)

            # Update dp_all
            old_dp_all = dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n:
                    val |= old_dp_all[k]
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all

            # Update dp_without
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 1)
                new_list[0] = 1
                for k in range(1, n + 1):
                    val = 0
                    if k <= n - 1:
                        val |= old_dp_w[k]
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list

            # For new element x, dp_without[x] is the old dp_all (set before adding x)
            new_dp_without[x] = list(old_dp_all)

            dp_without = new_dp_without
            n += 1

        return selected

    # --- Conflict Graph Model ---
    # An edge exists between two elements (or more generally, a constraint exists)
    # if they participate in a "mean violation" configuration.
    # The primary bottleneck for small subset sizes (k=2) is the 3-term arithmetic progression (3-AP).
    # If a, b, c are in S and b = (a+c)/2, then b is the mean of {a, c}.
    # This forms a conflict graph where an edge (u, v) means they cannot both be in S if there exists a w in S such that w is the mean of u, v.
    # Actually, the condition is: no element is the mean of any subset of >= 2 OTHER elements.
    # For k=2, this is exactly the 3-AP condition.
    # If we can find a large set with no 3-APs, it's a strong candidate.
    # However, we also need to check higher k.
    # Strategy: Use a greedy algorithm that prioritizes elements based on their "conflict degree" in the 3-AP graph.
    # Elements that form many 3-APs with others should be added later or avoided if possible.
    # Or, conversely, pick elements that are "safe" (low conflict) first.

    def build_conflict_graph():
        """
        Build a graph where an edge (u, v) exists if u and v are part of a 3-AP with some w in [1, 512].
        i.e., u+v = 2w for some w.
        This is a heuristic proxy for "how likely are u and v to cause a conflict with a third element".
        """
        # For each pair (u, v), check if (u+v) is even and (u+v)/2 is in [1, 512] and distinct from u, v.
        # If so, u and v are in a 3-AP.
        # We can count the number of such w's for each pair to weight the edge.
        # Actually, for the purpose of greedy selection, we just want to know if adding x creates a conflict.
        # The fast greedy already does this precisely.
        # The "conflict graph" idea here is to use it for *ordering* or *local search*.
        # Let's compute a "risk score" for each element: how many 3-APs does it participate in?
        # If an element is the mean of many pairs, it's risky to include.
        # If an element is an endpoint of many 3-APs, it's also risky.
        
        risk = [0] * (MAX_N + 1)
        for w in range(1, MAX_N + 1):
            # w is the mean. Pairs (u, v) such that (u+v)/2 = w => u+v = 2w.
            # u, v in [1, 512], u != v, u != w, v != w.
            for u in range(1, MAX_N + 1):
                v = 2 * w - u
                if 1 <= v <= MAX_N and v != u and v != w:
                    # u and v form a 3-AP with w
                    risk[u] += 1
                    risk[v] += 1
                    risk[w] += 1 # w is the mean, so it's "involved"
        
        return risk

    # --- Local Search / Improvement ---
    def improve_set(initial_set, time_limit=5.0):
        """
        Try to improve the set by swapping elements or adding/removing.
        This is a local search heuristic.
        """
        if not initial_set:
            return []
        
        current_set = set(initial_set)
        current_list = sorted(initial_set)
        best_set = set(current_set)
        best_len = len(current_set)
        
        # Precompute DP for the current set to speed up checks?
        # That's expensive. Let's just use the fast greedy check on the modified set.
        # But fast greedy is O(N^2 * K) per insertion.
        # For local search, we need to check if a swap (remove a, add b) is valid.
        # This requires checking the new set.
        
        # Given the time constraints and complexity, a simple randomized restart with better ordering
        # might be more effective than a complex local search.
        # Let's stick to the "conflict-aware ordering" for the greedy phase.
        
        return list(best_set)

    # --- Main Search Logic ---
    best_set = []
    best_len = 0

    # 1. Compute risk scores for ordering
    risk_scores = build_conflict_graph()
    
    # Sort candidates by risk score (ascending) to pick "safe" elements first
    # Or descending? If we pick low-risk elements first, we might avoid creating many 3-APs.
    # Let's try ascending risk first.
    candidates_by_risk = sorted(range(1, MAX_N + 1), key=lambda x: risk_scores[x])
    
    # Strategy 1: Greedy with Risk-Aware Ordering (Low Risk First)
    res = try_greedy(candidates_by_risk)
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Greedy with Risk-Aware Ordering (High Risk First)
    # Sometimes picking "hard" elements first forces the rest to be compatible.
    candidates_by_risk_desc = sorted(range(1, MAX_N + 1), key=lambda x: -risk_scores[x])
    if time.time() - start_time < deadline - 5:
        res = try_greedy(candidates_by_risk_desc)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Standard Ascending
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(1, MAX_N + 1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 4: Standard Descending
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(MAX_N, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 5: Randomized Restarts with Risk-Based Bias
    # Blend random order with risk score
    for i in range(100):
        if time.time() - start_time > deadline - 2:
            break
        if best_len >= MAX_SIZE:
            break
        
        random.seed(i * 98765 + 43210)
        
        # Create a weighted order: low risk elements have higher probability of being earlier
        # We can sort by (risk_score + random_noise)
        noise_factor = 100  # Scale of noise
        candidates = list(range(1, MAX_N + 1))
        candidates.sort(key=lambda x: risk_scores[x] + random.random() * noise_factor)
        
        res = try_greedy(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= MAX_SIZE:
                break

    # Strategy 6: Modular Orders (from inspiration)
    for mod in [7, 11, 13, 17, 19, 23, 29, 31]:
        if time.time() - start_time > deadline - 3:
            break
        buckets = [[] for _ in range(mod)]
        for x in range(1, MAX_N + 1):
            buckets[x % mod].append(x)
        order = []
        max_len = max(len(b) for b in buckets) if buckets else 0
        for i in range(max_len):
            for b in buckets:
                if i < len(b):
                    order.append(b[i])
        res = try_greedy(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= MAX_SIZE:
                break

    # Strategy 7: Pure Random Restarts
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 3:
        for i in range(50):
            if time.time() - start_time > deadline - 1:
                break
            if best_len >= MAX_SIZE:
                break
            
            random.seed(123456 + i * 7919)
            candidates = list(range(1, MAX_N + 1))
            random.shuffle(candidates)
            res = try_greedy(candidates)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)

    return best_set

# EVOLVE-BLOCK-END
