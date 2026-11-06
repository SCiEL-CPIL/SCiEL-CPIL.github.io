"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    
    Strategy:
    1. Use a greedy approach that prioritizes elements that block the most candidates.
    2. Maintain a set of 'forbidden' values (those that would break the Sidon property if added).
    3. At each step, among all currently addable elements, pick the one that maximizes
       the number of new forbidden elements created (to force early saturation).
    4. Use randomized restarts with different random seeds to explore different solutions.
    5. Return the smallest maximal Sidon set found within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 25.0  # Stay safely under 30s limit
    
    best_solution = None
    best_size = float('inf')
    
    def compute_pair_sums(S):
        """Compute all pair sums a+b where a<=b from set S."""
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            sums.add(2 * a)
            for b in L[i + 1:]:
                sums.add(a + b)
        return sums
    
    def is_sidon(S):
        """Check if S is a Sidon set (all pair sums distinct)."""
        sums = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in sums:
                return False
            sums.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_addable_elements(S, used_sums):
        """Get all elements in [1,N] not in S that can be added while preserving Sidon property."""
        addable = []
        for x in range(1, N + 1):
            if x in S:
                continue
            # Check if adding x would create duplicate sums
            new_sums = set()
            valid = True
            for a in S:
                s = a + x
                if s in used_sums or s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                if 2 * x in used_sums or 2 * x in new_sums:
                    valid = False
                else:
                    new_sums.add(2 * x)
            if valid:
                addable.append(x)
        return addable
    
    def count_newly_blocked(S, x, used_sums):
        """
        Count how many currently-addable elements would become non-addable if we add x.
        This measures how much 'saturation' adding x causes.
        """
        # First, get all currently addable elements
        addable = get_addable_elements(S, used_sums)
        if not addable:
            return 0
        
        # Now simulate adding x
        new_S = set(S) | {x}
        new_sums = used_sums | {a + x for a in S} | {2 * x}
        
        # Count how many of the previously addable elements are no longer addable
        blocked = 0
        for y in addable:
            if y == x:
                continue
            # Check if y is still addable after adding x
            valid = True
            temp_sums = set()
            for a in new_S:
                s = a + y
                if s in new_sums or s in temp_sums:
                    valid = False
                    break
                temp_sums.add(s)
            if valid:
                if 2 * y in new_sums or 2 * y in temp_sums:
                    valid = False
            if not valid:
                blocked += 1
        return blocked
    
    def greedy_maximality_driven(random_seed=0):
        """
        Greedy algorithm that picks elements to maximize newly blocked candidates.
        """
        rng = random.Random(random_seed)
        S = set()
        used_sums = set()
        
        # Start with a random initial element to diversify
        initial_pool = list(range(1, N + 1))
        rng.shuffle(initial_pool)
        
        # Try to build from scratch
        while True:
            # Get all addable elements
            addable = get_addable_elements(S, used_sums)
            if not addable:
                break
            
            # If time is running out, just pick randomly and finish
            if time.time() > deadline - 5.0:
                # Quick finish: just pick the first addable element repeatedly
                while True:
                    addable = get_addable_elements(S, used_sums)
                    if not addable:
                        break
                    x = addable[0]
                    S.add(x)
                    for a in list(S):
                        used_sums.add(a + x)
                    used_sums.add(2 * x)
                break
            
            # Score each addable element by how many new elements it blocks
            scored = []
            for x in addable:
                blocked = count_newly_blocked(S, x, used_sums)
                scored.append((blocked, x))
            
            # Pick the element that blocks the most (with some randomness for diversity)
            if scored:
                # Sort by blocked count descending, but add some randomness
                scored.sort(key=lambda t: (-t[0], rng.random()))
                # Pick from top few with some probability
                top_k = min(3, len(scored))
                if rng.random() < 0.7:
                    x = scored[0][1]  # Most blocking
                else:
                    x = scored[rng.randrange(top_k)][1]
                
                # Add x to S
                S.add(x)
                for a in list(S):
                    used_sums.add(a + x)
                used_sums.add(2 * x)
        
        return list(S)
    
    def is_maximal(S):
        """Check if S is inclusion-maximal (no element can be added)."""
        used_sums = compute_pair_sums(S)
        for x in range(1, N + 1):
            if x in S:
                continue
            # Check if x can be added
            valid = True
            for a in S:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
            if valid and 2 * x not in used_sums:
                return False  # Can add x, so not maximal
        return True
    
    # Run multiple randomized restarts
    num_restarts = 50
    for seed in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        solution = greedy_maximality_driven(seed)
        
        # Verify it's a valid Sidon set
        if not is_sidon(solution):
            continue
        
        # Check if it's maximal
        if not is_maximal(solution):
            # Try to make it maximal by greedily adding elements
            S = set(solution)
            used_sums = compute_pair_sums(S)
            changed = True
            while changed:
                changed = False
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    # Check if x can be added
                    valid = True
                    for a in S:
                        s = a + x
                        if s in used_sums:
                            valid = False
                            break
                    if valid and 2 * x not in used_sums:
                        S.add(x)
                        for a in list(S):
                            used_sums.add(a + x)
                        used_sums.add(2 * x)
                        changed = True
                        break
            solution = list(S)
        
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
            
            # Early termination if we find a very good solution
            if best_size <= 12:
                break
    
    # If no valid solution found, fall back to simple greedy
    if best_solution is None:
        S = set()
        used_sums = set()
        for x in range(1, N + 1):
            valid = True
            for a in S:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
            if valid and 2 * x not in used_sums:
                S.add(x)
                for a in list(S):
                    used_sums.add(a + x)
                used_sums.add(2 * x)
        best_solution = list(S)
    
    return best_solution
# EVOLVE-BLOCK-END
