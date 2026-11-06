"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _precompute_triples(N=180):
    """Precompute all forbidden triples (a, b, c) with a < b < c, distinct,
    satisfying 1/a = 1/b + 1/c, i.e., a = b*c/(b+c) is an integer < b."""
    triples = []
    # For each pair (b, c) with 1 <= b < c <= N:
    # a = b*c/(b+c) must be a positive integer and a < b (since a < b < c)
    # a = b*c/(b+c) => a*(b+c) = b*c => a*b + a*c = b*c
    # Rearranging: b*c - a*b - a*c = 0 => (b-a)*(c-a) = a^2
    # So for each a, we need factor pairs of a^2: (b-a) * (c-a) = a^2
    # with b-a > 0, c-a > 0, and b < c, and b > a, c > b
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d of a2 such that d < a2/d (to ensure b < c)
        # b = a + d, c = a + a2/d
        # Need b < c => d < a2/d => d^2 < a2 => d < a
        # Also need b > a => d > 0 (always true)
        # And c <= N
        for d in range(1, a):  # d < a ensures b < c
            if a2 % d == 0:
                e = a2 // d  # c - a
                b = a + d
                c = a + e
                if b < c and c <= N and b > a:
                    triples.append((a, b, c))
    return triples

def _check_valid(S, triples):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def _count_conflicts(S, triples):
    """Count number of forbidden triples fully contained in S."""
    S_set = set(S)
    count = 0
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin before 90s kill
    
    N = 180
    triples = _precompute_triples(N)
    Ntriples = len(triples)
    
    # Build adjacency: for each element, which triples it participates in
    # triples_by_element[x] = list of triple indices involving x
    triples_by_element = {x: [] for x in range(1, N + 1)}
    for i, (a, b, c) in enumerate(triples):
        triples_by_element[a].append(i)
        triples_by_element[b].append(i)
        triples_by_element[c].append(i)
    
    # Greedy approach: start with all elements, remove elements to break triples
    # Strategy: iterative local search / simulated annealing style
    
    # Start with a good initial solution: take all large elements
    # Elements > N/2 = 90: no triple can have two elements > 90 with the third also in range
    # Actually, for a < b < c with a = bc/(b+c), if b > 90 and c > 90, then a = bc/(b+c)
    # a < b, and a = bc/(b+c) < bc/(2*90) if b,c > 90... let's just check
    # For b,c > 90: a = bc/(b+c). Since b,c <= 180, a <= 180*180/181 ≈ 179.5, but a < b.
    # If b=91, c=92: a = 91*92/183 ≈ 45.9, not integer
    # The initial solution range(91,181) has 90 elements and is valid.
    
    best_S = set(range(91, 181))
    best_size = len(best_S)
    
    # Try to add elements from 1..90 to the set
    # For each candidate x in 1..90, check if adding it creates any triple
    # A triple involving x: x could be a, b, or c
    
    def can_add(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        for i in triples_by_element.get(x, []):
            a, b, c = triples[i]
            other1, other2 = None, None
            if x == a:
                other1, other2 = b, c
            elif x == b:
                other1, other2 = a, c
            else:
                other1, other2 = a, b
            if other1 in S and other2 in S:
                return False
        return True
    
    # Greedy: try to add elements from 1 to 90
    S = set(best_S)
    for x in range(1, 91):
        if can_add(x, S):
            S.add(x)
    
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
    
    # Now do local search: try to add more elements, or swap
    # Also try removing elements that are in many triples to free up space
    
    # Simulated annealing / local search
    random.seed(42)
    current_S = set(best_S)
    current_size = len(current_S)
    
    # Precompute for faster conflict checking
    # For each triple, track which elements are present
    # We'll use a different approach: for each element, maintain count of "active" triples
    
    def get_conflict_count(S):
        """Return list of (triple_index, [elements_in_S]) for all triples with >=1 element in S."""
        S_set = set(S)
        conflicts = []
        for i, (a, b, c) in enumerate(triples):
            in_S = [x for x in (a, b, c) if x in S_set]
            if len(in_S) >= 1:
                conflicts.append((i, in_S))
        return conflicts
    
    # Better approach: for local search, try adding one element at a time
    # If it creates conflicts, try removing one of the conflicting elements
    
    def try_add_and_remove(S, x):
        """Try to add x to S. If conflicts arise, try removing one element from each conflict."""
        if x in S:
            return S, 0
        if not can_add(x, S):
            return S, 0
        new_S = S | {x}
        return new_S, 1
    
    # Iterative improvement: repeatedly try to add the best candidate
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try adding each element not in S
        candidates = [x for x in range(1, N + 1) if x not in current_S]
        random.shuffle(candidates)
        for x in candidates:
            if (time.time() - start_time) > deadline:
                break
            new_S, gain = try_add_and_remove(current_S, x)
            if gain > 0:
                current_S = new_S
                improved = True
                if len(current_S) > best_size:
                    best_S = set(current_S)
                    best_size = len(current_S)
    
    # Try a different initial solution: random subset
    for trial in range(20):
        if (time.time() - start_time) > deadline - 5:
            break
        # Random initial set: take each element with some probability
        S = set()
        for x in range(1, N + 1):
            if random.random() < 0.5:
                S.add(x)
        # Remove conflicts greedily
        for (a, b, c) in triples:
            if a in S and b in S and c in S:
                # Remove the one with most conflicts
                counts = []
                for elem in (a, b, c):
                    cnt = len(triples_by_element.get(elem, []))
                    counts.append((cnt, elem))
                counts.sort(reverse=True)
                S.remove(counts[0][1])
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Try to improve
        improved = True
        while improved and (time.time() - start_time) < deadline - 3:
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if (time.time() - start_time) > deadline - 3:
                    break
                new_S, gain = try_add_and_remove(S, x)
                if gain > 0:
                    S = new_S
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
    
    # Final check and return
    result = sorted(best_S)
    assert len(result) > 0
    assert _check_valid(result, triples), "Result is not valid!"
    
    return result
# EVOLVE-BLOCK-END
