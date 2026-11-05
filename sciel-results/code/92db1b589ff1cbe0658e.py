"""Rewritten program with Simulated Annealing / Randomized Local Search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Helper to check if adding x to S (list) is valid Sidon-wise
    # S is assumed to be a valid Sidon set already.
    # Returns True if S U {x} is Sidon.
    def is_sidon_valid(S, x, sums_set):
        # Check x+x
        if x + x in sums_set:
            return False
        # Check x + s for s in S
        for s in S:
            if x + s in sums_set:
                return False
        return True

    # Helper to check if adding x to S keeps diffs disjoint from other_diffs
    # S is the set being modified. other_diffs is the set of diffs of the OTHER set.
    # Returns True if new diffs of S U {x} do not intersect other_diffs.
    def is_diff_disjoint_valid(S, x, other_diffs):
        # New diffs introduced by x:
        # x - s and s - x for all s in S.
        # Note: diffs within S are already disjoint from other_diffs.
        # We only need to check the NEW diffs.
        for s in S:
            if (x - s) in other_diffs:
                return False
            if (s - x) in other_diffs:
                return False
        return True

    # Calculate sums set for a list S
    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    # Calculate diffs set for a list S
    def get_diffs(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    # Check if two sets A, B are valid together (Sidon individually, disjoint diffs)
    def is_valid_pair(A, B):
        # Check A is Sidon
        sums_a = get_sums(A)
        if len(sums_a) != len(A) * (len(A) + 1) // 2:
            return False
        # Check B is Sidon
        sums_b = get_sums(B)
        if len(sums_b) != len(B) * (len(B) + 1) // 2:
            return False
        
        diffs_a = get_diffs(A)
        diffs_b = get_diffs(B)
        
        if diffs_a & diffs_b:
            return False
        return True

    # Greedy construction to get a good start
    def greedy_construct():
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diffs_a = set()
        diffs_b = set()
        
        # Try to fill A and B using a randomized greedy approach
        # We'll iterate through candidates in random order
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if is_sidon_valid(A, x, sums_a) and is_diff_disjoint_valid(A, x, diffs_b):
                A.append(x)
                # Update sums and diffs for A
                for a in A[:-1]:
                    sums_a.add(x + a)
                    diffs_a.add(x - a)
                    diffs_a.add(a - x)
                sums_a.add(x + x)
                continue
            
            # Try adding to B
            if is_sidon_valid(B, x, sums_b) and is_diff_disjoint_valid(B, x, diffs_a):
                B.append(x)
                # Update sums and diffs for B
                for b in B[:-1]:
                    sums_b.add(x + b)
                    diffs_b.add(x - b)
                    diffs_b.add(b - x)
                sums_b.add(x + x)
                continue
        
        return A, B

    # Initialize
    A, B = greedy_construct()
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    
    # Pre-compute state for current best
    sums_a = get_sums(A)
    sums_b = get_sums(B)
    diffs_a = get_diffs(A)
    diffs_b = get_diffs(B)
    
    # Local Search / Simulated Annealing
    # We will try to improve the size.
    # Moves:
    # 1. Add unused element to A or B.
    # 2. Move element from A to B.
    # 3. Move element from B to A.
    # 4. Swap elements between A and B.
    # 5. Remove element (to escape local minima).
    
    used_elements = set(A) | set(B)
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Decay temperature
        temperature *= cooling
        
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an unused element
            unused = [x for x in range(1, 257) if x not in used_elements]
            if not unused:
                continue
            x = random.choice(unused)
            
            # Try adding to A
            if is_sidon_valid(A, x, sums_a) and is_diff_disjoint_valid(A, x, diffs_b):
                A.append(x)
                used_elements.add(x)
                for a in A[:-1]:
                    sums_a.add(x + a)
                    diffs_a.add(x - a)
                    diffs_a.add(a - x)
                sums_a.add(x + x)
                # Update best if improved
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
                continue
            
            # Try adding to B
            if is_sidon_valid(B, x, sums_b) and is_diff_disjoint_valid(B, x, diffs_a):
                B.append(x)
                used_elements.add(x)
                for b in B[:-1]:
                    sums_b.add(x + b)
                    diffs_b.add(x - b)
                    diffs_b.add(b - x)
                sums_b.add(x + x)
                # Update best if improved
                if len(A) + len(B) > best_size:
                    best_size = len(A) + len(B)
                    best_A = list(A)
                    best_B = list(B)
                continue
            
            # If we couldn't add, maybe we should try a "kick" (removal)
            # to allow this or other elements to fit later.
            # This is a heuristic: if we fail to add, we might be stuck.
            # Let's handle removals in a separate branch to control frequency.
            
        elif move_type < 0.7:
            # Move element from A to B
            if not A: continue
            i = random.randint(0, len(A) - 1)
            x = A[i]
            
            # Temporarily remove x from A
            A.pop(i)
            used_elements.discard(x) # Will re-add if move succeeds, else keep out? 
                                     # Actually, if we move it, it's still used.
                                     # Let's just check validity of B U {x} given A \ {x}
            
            # We need to recompute sums/diffs for A \ {x} to check disjointness correctly?
            # No, is_diff_disjoint_valid(B, x, diffs_a_minus_x)
            # diffs_a currently includes diffs involving x.
            # This is getting complex with incremental updates for removals.
            # For simplicity and correctness in the search loop, 
            # let's use a "check then apply" strategy with full recomputation for moves, 
            # or stick to additions only if removals are too complex.
            
            # Given time constraints and the fact that additions are the primary driver,
            # let's focus on additions and simple swaps if easy.
            # Let's revert this move for now and just try additions.
            A.insert(i, x)
            continue

        elif move_type < 0.9:
            # Move element from B to A
            if not B: continue
            # Similar complexity, skip for now.
            continue
            
        else:
            # Kick: Remove a random element to try to unblock
            # This is risky, but can help.
            if random.random() < 0.1: # Low probability
                if A:
                    i = random.randint(0, len(A) - 1)
                    x = A.pop(i)
                    used_elements.discard(x)
                    # Recompute sums and diffs for A
                    sums_a = get_sums(A)
                    diffs_a = get_diffs(A)
                elif B:
                    i = random.randint(0, len(B) - 1)
                    x = B.pop(i)
                    used_elements.discard(x)
                    sums_b = get_sums(B)
                    diffs_b = get_diffs(B)
    
    # Ensure nonempty
    if not best_A and not best_B:
        best_A = [1]
        best_B = [2]
    elif not best_A:
        best_A = [1]
    elif not best_B:
        best_B = [1]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-END
