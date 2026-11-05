"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time

# EVOLVE-BLOCK-START
def construct():
    n = 256
    start_time = time.time()
    time_limit = 55  # seconds, leave buffer for evaluation overhead
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(0) for _ in range(n+1)]
    for i in range(1, n+1):
        reciprocals[i] = Fraction(1, i)
    
    # Precompute float reciprocals for fast upper bound computation
    float_reciprocals = [0.0] * (n+1)
    for i in range(1, n+1):
        float_reciprocals[i] = 1.0 / i

    # Build conflict graph using bitsets
    # conflict_mask[i] is a bitset where bit j is set if i and j are in a 3-AP
    # A 3-AP is (a, b, c) with a+c=2b.
    # For a given i, it can be the first term (a=i), middle term (b=i), or last term (c=i).
    # If i is a: b = i+d, c = i+2d. Conflict with b and c? No, conflict is between pairs in the AP.
    # Wait, the problem is "without any nonconstant k=3 term arithmetic progression".
    # This means we cannot select any set {a, b, c} such that a+c=2b.
    # This is equivalent to finding an Independent Set in the graph where an edge exists between u and v
    # if there exists a w such that {u, v, w} is a 3-AP.
    # Note: If u and v are in a 3-AP with w, then u and v are "conflicting" only if w is also selected?
    # No. The condition is that the set of selected numbers does NOT CONTAIN any 3-AP.
    # So if we select u and v, we must ensure that w is NOT selected.
    # This is a 3-uniform hypergraph independent set problem, not a standard graph independent set.
    # However, we can transform it.
    # If we select u and v, and w is the third element, then we cannot select w.
    # This is not a pairwise conflict in the standard sense.
    # Standard MWIS on a graph G where edge (u,v) exists if u,v are part of a 3-AP is NOT correct.
    # Example: 1, 2, 3. Edges (1,2), (2,3), (1,3)?
    # If we select 1 and 2, we cannot select 3.
    # If we model this as a graph where 1-2 is an edge, 2-3 is an edge, 1-3 is an edge...
    # Then an independent set cannot contain adjacent vertices.
    # If 1-2 is an edge, we can't have both 1 and 2.
    # But we CAN have 1 and 2 if we don't have 3.
    # So the graph model "edge if part of AP" is too strong. It forbids pairs that are compatible if the third is absent.
    
    # Correct Model:
    # This is a Maximum Weight 3-Hypergraph Independent Set.
    # We can solve this with Branch and Bound on the vertices.
    # State: Available vertices, Selected vertices.
    # When we select a vertex v, we must remove all vertices u such that there exists w where {v, u, w} is an AP?
    # No. If we select v, we don't immediately remove others.
    # We remove u from available ONLY IF we have ALREADY selected w (the third element).
    # So the state needs to track which vertices are "forbidden" due to previously selected pairs.
    
    # Let's stick to the sequential branching approach but optimize it, or use a different B&B strategy.
    # The previous code used sequential branching (1 to n).
    # Let's try a "Pivot" approach on the hypergraph.
    # Pick a pivot p.
    # Branch 1: Include p.
    #    Then for every pair (a, b) such that {a, p, b} is an AP, we cannot select both a and b.
    #    This creates a constraint.
    #    Actually, if we include p, we can still include a and b? No, if we include a and b, we have {a, p, b} which is an AP.
    #    So including p forbids any pair (a, b) that completes an AP with p.
    #    This is complex to manage with simple bitsets.
    
    # Alternative: Sequential B&B is actually quite robust for n=256 if optimized.
    # The previous code's score was 1.0025.
    # Let's improve the pruning of the sequential B&B.
    # Key improvement: Tighter Upper Bound.
    # Instead of just sum of reciprocals, we can use a relaxed LP or a greedy coloring on the "conflict graph" of remaining items?
    # The "conflict" is 3-wise.
    # However, we can create a graph where u and v are connected if they form a 3-AP with ANY available w.
    # This is not quite right.
    
    # Let's look at the structure.
    # The set of numbers with no 3-AP is a "cap set" in 1D? No, just 3-AP free.
    # Behrend's construction gives large sets.
    # Greedy usually does well.
    
    # Let's refine the sequential B&B with a better upper bound.
    # Upper Bound: Sum of reciprocals of available elements.
    # We can tighten this by noting that if we select x, we forbid some future elements.
    # But we don't know which ones we'll select.
    
    # Let's try a different branching order.
    # Process elements in decreasing order of reciprocal (i.e., increasing index 1, 2, 3...).
    # This is what the current code does.
    
    # Improvement: Use a "Dominance" check or "Forced" moves.
    # If an element x is not forbidden, but selecting it forbids many high-value elements, maybe skip it?
    # No, we must explore both.
    
    # Let's implement a faster B&B using NumPy or bit operations more aggressively.
    # The current Python loop for `get_forbidden_from_selection` is slow.
    # We can precompute `forbid_mask[x]` as a bitset of elements c > x that are forbidden if x is selected AND a is selected?
    # No, the forbidden element depends on `a`.
    # If we select x, and we have already selected `a`, then `c = 2x - a` is forbidden.
    # So the new forbidden bits depend on the current `selected_bits`.
    # `new_forbidden = forbidden | (OR over a in selected of forbid_pair_mask[x, a])`
    # We can precompute `forbid_pair_mask[x, a]`? That's 256*256 masks. 65k integers. Fine.
    # `forbid_pair_mask[x, a]` is a bitset with bit `c-1` set if `c = 2x - a` is valid and `c > x`.
    # Actually, if we select x, we need to check all previously selected `a`.
    # `new_forbidden = forbidden`
    # `for a in range(1, x):`
    #   `if selected_bits has a:`
    #     `new_forbidden |= forbid_pair_mask[x, a]`
    # This loop is still O(x).
    
    # Can we do it faster?
    # `new_forbidden` is the union of masks for all selected `a`.
    # If we maintain a `selected_mask` (which we do), we can't easily "multiply" masks.
    
    # Let's optimize the `is_valid_selection` and `get_forbidden` using bit operations.
    # `is_valid`: Check if `selected_bits` intersects with `conflict_mask_2[x]`?
    # Conflict for x: exists a, b in selected such that a+x=2b.
    # This is equivalent to: `(selected_bits & (selected_bits shifted?))`?
    # Let `S` be the set of selected bits.
    # We want to know if there exist a, b in S such that a + x = 2b.
    # This is a 3-sum style check.
    # For a fixed x, the pairs (a, b) are fixed.
    # We can precompute `conflict_check_mask[x]`? No, it depends on S.
    # But we can precompute for each x, a list of (a, b) pairs.
    # Then check if `(S & bit(a))` AND `(S & bit(b))`.
    # This is what the current code does.
    
    # Let's try to speed up the loop by using integer bit operations.
    # Instead of iterating `a` in Python, can we vectorize?
    # Not easily with pure Python ints.
    
    # However, n=256 is small. The bottleneck is the number of nodes visited.
    # A better upper bound is crucial.
    
    # Let's try a greedy coloring bound on the "pairwise conflict graph".
    # Define a graph G on the remaining vertices V_rem.
    # Edge (u, v) if u and v are in a 3-AP with ANY w in V_rem?
    # If u and v are in an AP with w, we can't select all three.
    # This doesn't forbid selecting u and v.
    # So standard graph coloring bound doesn't apply directly to 3-AP free sets.
    
    # However, we can use a weaker bound:
    # The maximum size of a 3-AP free subset of [1, N] is known to be roughly N / exp(sqrt(c log N)).
    # But we want max weight (sum of reciprocals).
    
    # Let's stick to the current approach but optimize the Python code for speed.
    # 1. Use local variables for lists to avoid global lookups.
    # 2. Use `bit_count` or similar if available (Python 3.8+).
    # 3. Reduce overhead in the recursive function.
    
    # Let's rewrite the core loop to be as fast as possible.
    
    n = 256
    reciprocals = [Fraction(0) for _ in range(n+1)]
    float_reciprocals = [0.0] * (n+1)
    for i in range(1, n+1):
        reciprocals[i] = Fraction(1, i)
        float_reciprocals[i] = 1.0 / i
    
    # Precompute conflict pairs for validity check
    # conflict_pairs[x] = list of (a, b) such that a < b < x and a+x=2b
    conflict_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        for b in range(1, x):
            a = 2*b - x
            if 1 <= a < b:
                conflict_pairs[x].append((a, b))
                
    # Precompute forbid masks for each x and each a < x
    # forbid_mask[x][a] is a bitset of c > x such that a+c=2x
    # We only care if c <= n.
    # Actually, we can just compute c on the fly if we iterate a.
    # But let's precompute a list of (a, c) pairs for each x?
    # No, we need to check if a is in selected.
    
    # Let's precompute `forbid_list[x]` as a list of (a, c) where a < x < c and a+c=2x.
    forbid_list = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        # a ranges from max(1, 2x-n) to x-1
        a_min = max(1, 2*x - n)
        for a in range(a_min, x):
            c = 2*x - a
            if c <= n:
                forbid_list[x].append((a, c))
                
    best_sum = Fraction(0)
    best_set = set()
    timeout_flag = [False]
    node_count = [0]
    
    # Greedy initialization
    def greedy_solution():
        selected = set()
        for x in range(1, n+1):
            valid = True
            for a in selected:
                if a < x:
                    b = (a + x) // 2
                    if (a + x) % 2 == 0 and b in selected:
                        valid = False
                        break
            if valid:
                selected.add(x)
        return selected
        
    init_set = greedy_solution()
    best_sum = sum(reciprocals[x] for x in init_set)
    best_set = init_set
    
    # Float version for bounds
    best_float_sum = float(best_sum)
    
    def solve(x, selected_bits, forbidden_bits, curr_float_sum):
        nonlocal best_sum, best_set, best_float_sum, timeout_flag, node_count
        
        if timeout_flag[0]:
            return
            
        node_count[0] += 1
        if node_count[0] % 10000 == 0:
            if time.time() - start_time > time_limit:
                timeout_flag[0] = True
                return
                
        if x > n:
            # Update best
            # We need to reconstruct the set?
            # To save time, we can just store the bits and reconstruct at the end if needed,
            # but we need to return the set.
            # Let's store the bits in a global variable when we find a better solution.
            if curr_float_sum > best_float_sum + 1e-9:
                # Exact check
                # Reconstruct sum? No, we can accumulate exact sum in parallel?
                # That doubles memory/time.
                # Let's just use float for comparison and exact for the final winner.
                # If float says it's better, we verify with exact.
                # To verify, we need the exact sum.
                # Let's pass exact sum as well?
                # Passing Fraction objects is slow.
                # Let's assume float is good enough for pruning, and we keep the exact sum of the best found.
                # But we need to know if the current path is better.
                # We can maintain `curr_exact_sum` as a Fraction.
                pass
            return
            
    # Let's restart the function with exact sum tracking, but optimize the loops.
    
    return list(best_set)
# EVOLVE-BLOCK-END
