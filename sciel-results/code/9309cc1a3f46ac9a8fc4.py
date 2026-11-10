"""
Maximum subset of {1..192} such that no element divides the sum of any 3 distinct other elements.
Uses a backtracking search with efficient conflict precomputation and branch-and-bound.
"""
import time
import random
import sys

sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0

N = 192
MAX_K = 48


def _precompute():
    """
    Precompute conflict information.
    
    For each candidate value d in 1..192, precompute for each pair (a, b) with a != b,
    the set of residues r such that (a + b + r) % d == 0, i.e., r ≡ (-a-b) mod d.
    
    Then during search, when we have a current set S and want to add d, we need to check:
    1. d doesn't divide sum of any 3 in S: for all triples (x,y,z) in S, (x+y+z)%d != 0
    2. For each a in S, a doesn't divide (d+x+y) for any distinct x,y in S\{a}
    
    We'll use bitsets for efficiency.
    """
    # For each d, and each residue class mod d, track which numbers have that residue
    # Actually, let's precompute a more useful structure.
    
    # divisors[d] = list of (a,b) pairs such that we need to check if there's a c in S
    # with (a+b+c) % d == 0. But that's too many.
    
    # Alternative: For each d, precompute for each pair of residues (r1, r2) mod d,
    # whether r1+r2+0 ≡ 0 (mod d) etc. Actually, let's just be direct but use numpy.
    
    # For the check "does d divide sum of any 3 in S":
    # We can precompute for each d, an array residue_count[d][r] = number of elements in S with value % d == r
    # Then d divides sum of 3 iff there exist r1,r2,r3 (not necessarily distinct residues,
    # but corresponding to distinct elements) such that r1+r2+r3 ≡ 0 mod d.
    
    # This is complex. Let's use a simpler but still fast approach:
    # Precompute for each d, a 2D array pair_sum_res[d][r1][r2] = (r1+r2) % d
    # Then check if 0 is achievable with 3 distinct elements.
    
    # Actually, the simplest fast check:
    # For adding d to S:
    #   Check 1: No triple in S sums to multiple of d
    #   Check 2: For each a in S, no pair (x,y) in S\{a} such that a | (d+x+y)
    
    # Precompute for each d: divisors_info[d] = for each a in 1..192, the target residue
    #   t = (-d) % a, so we need x+y ≡ t (mod a)
    
    # Let's precompute pair sums efficiently using sorting or residues.
    
    # I'll use a different strategy: precompute for each d, all (a,b) pairs from [1,192]
    # where a != b, and store the required residue for c: c ≡ (-a-b) mod d.
    # Then when checking if d can be added, for each a in S, check if there exists b in S\{a}
    # such that the required residue for c is present in S (as some element c not equal to a or b).
    
    # This is still O(|S|^2) per addition check in the worst case.
    
    # Let's try yet another approach: use Python's built-in big integers as bitsets.
    # Represent the selected set as a bitset.
    
    # Precompute: for each d in 1..192, and for each possible triple-sum residue,
    # but that's too much.
    
    # Let's just implement a reasonably fast check and rely on good search ordering.
    
    # Precompute divisors of each number up to 3*192=576
    # divisors[s] = list of divisors of s
    divisors = [[] for _ in range(3 * N + 1)]
    for d in range(1, N + 1):
        for multiple in range(d, 3 * N + 1, d):
            divisors[multiple].append(d)
    
    # For each triple (a,b,c) of distinct values in 1..192, precompute which d's conflict
    # i.e., d | (a+b+c) and d not in {a,b,c}
    # But there are C(192,3) ≈ 1.16 million triples, too many to store explicitly.
    
    # Let's use a hybrid: fast incremental check with caching where possible.
    
    return divisors


def _can_add(d, selected, selected_set):
    """
    Check if d can be added to selected without violating constraints.
    selected is a list, selected_set is a set for O(1) lookup.
    
    Returns True if valid to add.
    """
    n = len(selected)
    
    # Check 1: d must not divide sum of any 3 distinct elements in selected
    if n >= 3:
        # Use residue counting: count how many elements have each residue mod d
        if d <= 192:
            # Count residues
            res_count = [0] * d
            for x in selected:
                res_count[x % d] += 1
            
            # Now check if there exist 3 distinct elements (with indices) whose sum ≡ 0 mod d
            # We need to be careful: distinct elements, not distinct residues.
            # A residue r can contribute at most res_count[r] elements.
            
            # Try all pairs of residues (r1, r2) and see if r3 = (-r1-r2) % d has enough count
            # But we need 3 DISTINCT elements. If r1=r2=r3, need count >= 3.
            # If two are equal, say r1=r2!=r3, need count[r1]>=2 and count[r3]>=1.
            # If all different, need count[r1]>=1, count[r2]>=1, count[r3]>=1.
            
            found = False
            for r1 in range(d):
                if res_count[r1] == 0:
                    continue
                for r2 in range(r1, d):
                    if res_count[r2] == 0:
                        continue
                    r3 = (-r1 - r2) % d
                    if r3 < r2:
                        continue
                    # Now we have r1 <= r2 <= r3
                    if r1 == r2 == r3:
                        if res_count[r1] >= 3:
                            found = True
                            break
                    elif r1 == r2:
                        if res_count[r1] >= 2 and res_count[r3] >= 1:
                            found = True
                            break
                    elif r2 == r3:
                        if res_count[r1] >= 1 and res_count[r2] >= 2:
                            found = True
                            break
                    else:
                        if res_count[r1] >= 1 and res_count[r2] >= 1 and res_count[r3] >= 1:
                            found = True
                            break
                if found:
                    break
            if found:
                return False
    
    # Check 2: For each a in selected, a must not divide (d + x + y) for distinct x,y in selected\{a}
    for idx_a, a in enumerate(selected):
        target = (-d) % a
        # Need distinct x, y from selected \ {a} with (x + y) % a == target
        # Use residue approach
        res_seen = set()
        for i in range(n):
            if i == idx_a:
                continue
            x = selected[i]
            xr = x % a
            needed = (target - xr) % a
            if needed in res_seen:
                return False
            res_seen.add(xr)
    
    return True


def _backtrack_search(candidates, current_set, current_set_set, best, best_set, time_limit, deadline, depth=0):
    """
    Backtracking search to find maximum valid set.
    candidates: list of remaining candidate values to consider
    current_set: list of currently selected values
    current_set_set: set of currently selected values
    best: best size found so far
    best_set: best set found so far
    """
    elapsed = time.time() - _START_TIME
    if elapsed > deadline:
        return best, best_set
    
    if len(current_set) >= best:
        best = len(current_set)
        best_set = list(current_set)
    
    if not candidates:
        return best, best_set
    
    # Upper bound: current size + remaining candidates
    if len(current_set) + len(candidates) <= best:
        return best, best_set
    
    for i, d in enumerate(candidates):
        if time.time() - _START_TIME > deadline:
            return best, best_set
        
        if _can_add(d, current_set, current_set_set):
            new_set = current_set + [d]
            new_set_set = current_set_set | {d}
            remaining = candidates[i+1:]
            best, best_set = _backtrack_search(
                remaining, new_set, new_set_set, best, best_set, time_limit, deadline, depth+1
            )
    
    return best, best_set


def _greedy_construct(order):
    """Greedy construction with given order."""
    selected = []
    selected_set = set()
    for c in order:
        if _can_add(c, selected, selected_set):
            selected.append(c)
            selected_set.add(c)
    return selected


def _local_search(initial_set, max_time=10.0):
    """Local search with add moves."""
    start = time.time()
    best_set = list(initial_set)
    best_set_set = set(best_set)
    
    while time.time() - start < max_time:
        improved = False
        for c in range(N, 0, -1):
            if c in best_set_set:
                continue
            if _can_add(c, best_set, best_set_set):
                best_set.append(c)
                best_set_set.add(c)
                improved = True
                break
        if not improved:
            # Try swaps
            for i in range(len(best_set)):
                for c in range(N, 0, -1):
                    if c in best_set_set:
                        continue
                    reduced = [x for j, x in enumerate(best_set) if j != i]
                    reduced_set = best_set_set - {best_set[i]}
                    if _can_add(c, reduced, reduced_set):
                        new_set = reduced + [c]
                        best_set = new_set
                        best_set_set = set(best_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_set


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    divisors = _precompute()
    
    best_set = []
    best_size = 0
    
    # Multiple greedy starts
    random.seed(123)
    
    # Descending
    s1 = _greedy_construct(list(range(N, 0, -1)))
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # Ascending
    s2 = _greedy_construct(list(range(1, N + 1)))
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    # Random orders
    for i in range(30):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
    
    # Local search
    if time.time() < deadline - 5:
        improved = _local_search(best_set, max_time=min(20.0, deadline - time.time() - 3))
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    # Backtracking from a partial set to try to improve
    if time.time() < deadline - 3:
        # Try backtracking with the best set as starting point
        # Sort candidates by some heuristic
        candidates = [c for c in range(N, 0, -1) if c not in set(best_set)]
        
        # Use a simpler backtracking: try to add one more element
        for c in candidates:
            if time.time() > deadline - 2:
                break
            if _can_add(c, best_set, set(best_set)):
                new_set = best_set + [c]
                if len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(best_set)
    
    # Final safety: ensure valid
    if len(best_set) > 0:
        # Quick validation
        valid = True
        n = len(best_set)
        for i in range(n):
            d = best_set[i]
            others = [best_set[j] for j in range(n) if j != i]
            m = len(others)
            for x in range(m):
                for y in range(x + 1, m):
                    for z in range(y + 1, m):
                        if (others[x] + others[y] + others[z]) % d == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to a small valid set
            best_set = [192, 191, 190]
    
    return best_set
