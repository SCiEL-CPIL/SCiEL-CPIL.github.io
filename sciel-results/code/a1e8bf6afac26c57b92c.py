"""Improved program: branch-and-bound on hypergraph independence."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Returns a list of distinct positive integers <= 192 (max 48) such that
    no selected integer divides the sum of any 3 distinct OTHER selected integers.
    Maximizes cardinality.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute conflict structure:
    # For each pair (b,c), precompute which a values are "bad" (a divides b+c+some third)
    # Actually, we need: for a given set S and candidate x, check if adding x creates any violation.
    # A violation occurs if:
    #   1. For some a in S, there exist b,c in S\{a} such that (b+c+x) % a == 0
    #   2. For some triple (b,c,d) in S, (b+c+d) % x == 0
    #
    # Precompute for each a, and for each residue pair, the count of pairs (b,c) with b%c == r1, c%c == r2
    # such that (b+c) % a == some value.
    #
    # Better approach: Precompute a 2D array conflict[a][x] = list of pairs (b,c) that would cause
    # a violation if both b,c are in S and x is added. But this is too large.
    #
    # Alternative: For each a in [1..N], precompute for each residue r in [0..a-1],
    # the list of numbers in [1..N] with that residue mod a.
    # Then when checking if x can be added to S, for each a in S, we check if there exist
    # b,c in S\{a} such that (b+c) % a == (-x) % a.
    
    # Precompute residues: for each a in 1..N, for each residue r in 0..a-1,
    # store the list of numbers in 1..N with that residue.
    residues = {}
    for a in range(1, N + 1):
        residues[a] = {}
        for r in range(a):
            residues[a][r] = []
        for v in range(1, N + 1):
            r = v % a
            residues[a][r].append(v)
    
    # Precompute triple sums mod x for fast checking
    # For each x in 1..N, precompute which triples (b,c,d) have (b+c+d) % x == 0
    # This is too expensive. Instead, we'll use a different approach.
    
    def can_add(S_set, x):
        """Check if x can be added to set S (represented as a set of values)."""
        # Check condition 1: for each a in S, no pair (b,c) in S\{a} with (b+c+x) % a == 0
        for a in S_set:
            if a == x:
                continue
            target = (-x) % a
            # Count how many pairs (b,c) in S\{a} have (b+c) % a == target
            # We need at least one such pair where b != c and b,c != a
            # Use residue counting
            residue_count = {}
            for v in S_set:
                if v == a:
                    continue
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            # Count pairs (b,c) with b != c, b,c != a, (b+c) % a == target
            # Iterate over residues
            found = False
            for r1 in residue_count:
                r2 = (target - r1) % a
                if r2 not in residue_count:
                    continue
                cnt1 = residue_count[r1]
                cnt2 = residue_count[r2]
                if r1 == r2:
                    # Need at least 2 distinct elements with this residue
                    if cnt1 >= 2:
                        found = True
                        break
                else:
                    # Need at least 1 from each residue
                    if cnt1 >= 1 and cnt2 >= 1:
                        found = True
                        break
            if found:
                return False
        
        # Check condition 2: no triple (b,c,d) in S with (b+c+d) % x == 0
        if len(S_set) >= 3:
            # For each pair (b,c) in S, check if there exists d in S\{b,c} with (b+c+d) % x == 0
            S_list = list(S_set)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    b = S_list[i]
                    c = S_list[j]
                    target = (-b - c) % x
                    # Check if any d in S\{b,c} has d % x == target
                    for d in S_list:
                        if d == b or d == c:
                            continue
                        if d % x == target:
                            return False
        return True
    
    def is_valid_full(S_list):
        """Full validation of set S."""
        S_set = set(S_list)
        for a in S_set:
            others = [v for v in S_set if v != a]
            if len(others) < 3:
                continue
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            for i in range(len(others)):
                b = others[i]
                for j in range(i + 1, len(others)):
                    c = others[j]
                    target = (-b - c) % a
                    cnt = residue_count.get(target, 0)
                    if b % a == target:
                        cnt -= 1
                    if c % a == target:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    def greedy_construct(order):
        """Greedy construction: add numbers in given order if valid."""
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S_set, x):
                S.append(x)
                S_set.add(x)
            if len(S) >= MAX_K:
                break
        return S
    
    def local_search(S, remaining_time):
        """Local search: try to improve by adding or swapping elements."""
        S_set = set(S)
        
        while time.time() - start_time < remaining_time:
            improved = False
            
            # Try to add any number not in S
            for x in range(N, 0, -1):
                if x in S_set:
                    continue
                if can_add(S_set, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap moves
            swapped = False
            for i, x in enumerate(list(S)):
                if time.time() - start_time > remaining_time:
                    break
                S_without = [v for j, v in enumerate(S) if j != i]
                S_without_set = set(S_without)
                for y in range(N, 0, -1):
                    if y in S_without_set or y == x:
                        continue
                    if can_add(S_without_set, y):
                        S_candidate = S_without + [y]
                        S_candidate_set = set(S_candidate)
                        extra_added = 0
                        for z in range(N, 0, -1):
                            if z in S_candidate_set:
                                continue
                            if can_add(S_candidate_set, z):
                                S_candidate.append(z)
                                S_candidate_set.add(z)
                                extra_added += 1
                                if len(S_candidate) >= MAX_K:
                                    break
                        if extra_added > 0:
                            S = S_candidate
                            S_set = set(S)
                            improved = True
                            swapped = True
                            break
                if swapped:
                    break
            
            if not improved:
                break
        
        return S
    
    # Phase 1: Greedy construction with multiple orders
    best_S = []
    best_len = 0
    
    # Descending order
    S1 = greedy_construct(range(N, 0, -1))
    if len(S1) > best_len:
        best_S = S1
        best_len = len(S1)
    
    # Ascending order
    if time.time() - start_time < DEADLINE:
        S2 = greedy_construct(range(1, N + 1))
        if len(S2) > best_len:
            best_S = S2
            best_len = len(S2)
    
    # Random orders
    random.seed(42)
    for trial in range(300):
        if time.time() - start_time > DEADLINE - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = greedy_construct(order)
        if len(S3) > best_len:
            best_S = S3
            best_len = len(S3)
    
    # Phase 2: Local search on best solution
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE - 5)
    
    # Phase 3: More local search with different starting points
    if time.time() - start_time < DEADLINE - 10:
        for trial in range(50):
            if time.time() - start_time > DEADLINE - 10:
                break
            order = list(range(1, N + 1))
            random.shuffle(order)
            S = greedy_construct(order)
            if len(S) >= best_len:
                S = local_search(S, DEADLINE - 15)
                if len(S) > best_len:
                    best_S = S
                    best_len = len(S)
    
    # Phase 4: Final local search
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE - 2)
    
    # Final verification
    if not is_valid_full(best_S):
        best_S = greedy_construct(range(N, 0, -1))
    
    return best_S

# EVOLVE-BLOCK-END
