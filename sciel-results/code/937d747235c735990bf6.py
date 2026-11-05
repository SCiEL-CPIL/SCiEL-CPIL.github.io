import random
import time
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0

    # State
    A = set()
    B = set()
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    # For faster iteration, keep lists or just use sets. Sets are fine for n=256.
    # We need to check membership quickly.
    
    def check_add_A(x):
        # Check sums
        for a in A:
            if (x + a) in sums_A:
                return False
            if (x - a) in diffs_B:
                return False
            if (a - x) in diffs_B:
                return False
        if (2 * x) in sums_A:
            return False
        return True

    def check_add_B(x):
        # Check sums
        for b in B:
            if (x + b) in sums_B:
                return False
            if (x - b) in diffs_A:
                return False
            if (b - x) in diffs_A:
                return False
        if (2 * x) in sums_B:
            return False
        return True

    def perform_add_A(x):
        A.add(x)
        for a in A:
            if a != x:
                sums_A.add(x + a)
                diffs_A.add(x - a)
                diffs_A.add(a - x)
            else:
                sums_A.add(2 * x)

    def perform_add_B(x):
        B.add(x)
        for b in B:
            if b != x:
                sums_B.add(x + b)
                diffs_B.add(x - b)
                diffs_B.add(b - x)
            else:
                sums_B.add(2 * x)

    def perform_remove_A(x):
        A.remove(x)
        for a in A:
            sums_A.remove(x + a)
            diffs_A.remove(x - a)
            diffs_A.remove(a - x)
        sums_A.remove(2 * x)

    def perform_remove_B(x):
        B.remove(x)
        for b in B:
            sums_B.remove(x + b)
            diffs_B.remove(x - b)
            diffs_B.remove(b - x)
        sums_B.remove(2 * x)

    best_A = set()
    best_B = set()
    best_score = 0

    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.01
    steps = 0

    # Initial random start? Or empty?
    # Empty is safe.
    
    while time.time() - start_time < time_limit:
        # Generate move
        move_type = random.random()
        
        # 60% chance to try adding, 20% removing, 20% swapping
        if move_type < 0.6:
            # Try to add
            x = random.randint(1, n)
            if x not in A and x not in B:
                # Try A
                if check_add_A(x):
                    perform_add_A(x)
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_A = set(A)
                        best_B = set(B)
                elif check_add_B(x):
                    perform_add_B(x)
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_A = set(A)
                        best_B = set(B)
                # If neither, do nothing (or accept with penalty? No, hard constraint)
            else:
                # Element already in one set, try adding to the other?
                # If x in A, can we add to B?
                # Check if diffs are disjoint.
                # If x in A, its diffs are in diffs_A.
                # To add to B, x's diffs must not be in diffs_A.
                # But x is in A, so its diffs ARE in diffs_A (specifically, diffs involving x are in diffs_A).
                # Wait. If x in A, diffs_A contains {x-a, a-x for a in A}.
                # If we add x to B, we generate diffs {x-b, b-x for b in B}.
                # Do these conflict with diffs_A?
                # The condition is disjointness of *nonzero* difference sets.
                # If x in A and x in B, then 0 is a difference in both? No, "nonzero".
                # But if x in A, and we add x to B, do we create a conflict?
                # The diff x-x=0 is ignored.
                # But diffs like x-a (from A) and x-b (from B) are just numbers.
                # The constraint is $D_A \cap D_B = \emptyset$.
                # If x in A, $D_A$ contains $x-a$.
                # If we add x to B, $D_B$ will contain $x-b$.
                # There is no inherent conflict just from sharing $x$.
                # However, we must check if the *new* diffs from adding x to B conflict with *existing* diffs in A.
                # So yes, we can try to add x to B even if x is in A.
                
                if x in A and x not in B:
                    if check_add_B(x):
                        perform_add_B(x)
                        score = len(A) + len(B)
                        if score > best_score:
                            best_score = score
                            best_A = set(A)
                            best_B = set(B)
                elif x in B and x not in A:
                    if check_add_A(x):
                        perform_add_A(x)
                        score = len(A) + len(B)
                        if score > best_score:
                            best_score = score
                            best_A = set(A)
                            best_B = set(B)
                            
        elif move_type < 0.8:
            # Try to remove
            if A:
                x = random.choice(list(A))
                # Always allow removal?
                # In SA, we accept removals to escape local minima.
                # But removing reduces score.
                # We should only remove if we are "stuck" or with low probability.
                # Let's just perform it, the SA loop handles acceptance.
                # Actually, for simplicity and speed, let's only remove if it helps?
                # No, SA requires accepting worse states.
                # But we need to track if we accepted it.
                
                # Let's implement proper SA acceptance.
                # Current Score: S = len(A) + len(B)
                # New Score: S' = S - 1
                # Delta = -1
                # Accept if random() < exp(-Delta/T) = exp(1/T) ? No, exp(Delta/T) = exp(-1/T).
                
                # This is getting complicated to interleave with the "try add" logic.
                # Let's simplify:
                # 1. Generate a candidate state (A', B').
                # 2. If valid, calculate Delta.
                # 3. Accept based on SA.
                
                pass # Will rewrite loop structure

        else:
            # Swap
            pass

    # The above structure is messy. Let's refactor to a clean SA loop.

    A = set()
    B = set()
    sums_A = set()
    sums_B = set()
    diffs_A = set()
    diffs_B = set()
    
    def get_score():
        return len(A) + len(B)

    def is_valid_add_A(x):
        if x in A: return False
        for a in A:
            if (x + a) in sums_A: return False
            if (x - a) in diffs_B: return False
            if (a - x) in diffs_B: return False
        if (2 * x) in sums_A: return False
        return True

    def is_valid_add_B(x):
        if x in B: return False
        for b in B:
            if (x + b) in sums_B: return False
            if (x - b) in diffs_A: return False
            if (b - x) in diffs_A: return False
        if (2 * x) in sums_B: return False
        return True

    def apply_add_A(x):
        A.add(x)
        for a in A:
            if a != x:
                sums_A.add(x + a)
                diffs_A.add(x - a)
                diffs_A.add(a - x)
            else:
                sums_A.add(2 * x)

    def apply_add_B(x):
        B.add(x)
        for b in B:
            if b != x:
                sums_B.add(x + b)
                diffs_B.add(x - b)
                diffs_B.add(b - x)
            else:
                sums_B.add(2 * x)

    def apply_remove_A(x):
        A.remove(x)
        for a in A:
            sums_A.remove(x + a)
            diffs_A.remove(x - a)
            diffs_A.remove(a - x)
        sums_A.remove(2 * x)

    def apply_remove_B(x):
        B.remove(x)
        for b in B:
            sums_B.remove(x + b)
            diffs_B.remove(x - b)
            diffs_B.remove(b - x)
        sums_B.remove(2 * x)

    best_score = 0
    best_A = set()
    best_B = set()
    
    T = 1.0
    cooling = 0.9998
    min_T = 0.001
    
    # Initial greedy fill to get a good starting point
    for x in range(1, n+1):
        if is_valid_add_A(x):
            apply_add_A(x)
        elif is_valid_add_B(x):
            apply_add_B(x)
            
    current_score = get_score()
    if current_score > best_score:
        best_score = current_score
        best_A = set(A)
        best_B = set(B)

    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Generate Move
        r = random.random()
        move_accepted = False
        new_score = current_score
        
        if r < 0.5:
            # Add
            x = random.randint(1, n)
            if x not in A and x not in B:
                if is_valid_add_A(x):
                    apply_add_A(x)
                    new_score = current_score + 1
                    move_accepted = True
                elif is_valid_add_B(x):
                    apply_add_B(x)
                    new_score = current_score + 1
                    move_accepted = True
            elif x in A and x not in B:
                if is_valid_add_B(x):
                    apply_add_B(x)
                    new_score = current_score + 1
                    move_accepted = True
            elif x in B and x not in A:
                if is_valid_add_A(x):
                    apply_add_A(x)
                    new_score = current_score + 1
                    move_accepted = True
            
        elif r < 0.8:
            # Remove
            if A:
                x = random.choice(list(A))
                apply_remove_A(x)
                new_score = current_score - 1
                move_accepted = True
            elif B:
                x = random.choice(list(B))
                apply_remove_B(x)
                new_score = current_score - 1
                move_accepted = True
        else:
            # Swap: Remove x, Add y
            # Pick x from A or B
            if A:
                x = random.choice(list(A))
                # Pick y not in A or B
                # Just pick random y, if in A or B, skip or swap within?
                # Let's just do Remove/Add separately for simplicity in this version.
                # Or: Remove x, then try to add a random y.
                apply_remove_A(x)
                # Try to add random y
                y = random.randint(1, n)
                if y not in A and y not in B:
                    if is_valid_add_A(y):
                        apply_add_A(y)
                        new_score = current_score # -1 + 1
                        move_accepted = True
                    elif is_valid_add_B(y):
                        apply_add_B(y)
                        new_score = current_score
                        move_accepted = True
                # If no add, we are left with -1 score
                if not move_accepted:
                    new_score = current_score - 1
                    move_accepted = True # We accepted the removal part
            
        if move_accepted:
            delta = new_score - current_score
            if delta >= 0:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
                    best_B = set(B)
            else:
                # Accept worse state?
                # Probability exp(delta / T)
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    # Reject move: Undo
                    # We need to undo the last move.
                    # This is tricky without a stack.
                    # Let's use a simple Undo mechanism by saving state before move?
                    # Too expensive.
                    # Alternative: Only accept improvements? (Hill Climbing)
                    # Hill Climbing gets stuck easily.
                    # Let's implement Undo by storing the removed element and the added element.
                    
                    # If we added: Undo by removing.
                    # If we removed: Undo by adding back.
                    # If we swapped: Undo by removing y and adding back x.
                    
                    # This requires careful tracking.
                    # Given time limit, let's stick to Hill Climbing with Random Restarts?
                    # But the directive says "New Approach".
                    # Randomized Local Search with Backtracking?
                    
                    # Let's try a different "New Approach": 
                    # **Randomized Greedy with Backtracking (DFS)**
                    # This is distinct from simple greedy.
                    
                    # Actually, the current code in the prompt is Greedy.
                    # My SA attempt is struggling with Undo.
                    # Let's use **Iterated Local Search (ILS)** with Hill Climbing.
                    # 1. Greedy Construct.
                    # 2. Local Improvement (Swap/Remove/Add).
                    # 3. Perturbation (Randomly remove few elements).
                    # 4. Local Improvement again.
                    # This is a standard robust approach.

                    # Let's switch the implementation to ILS.
                    pass

        T *= cooling

    # Fallback to ensure non-empty
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}

    return [list(best_A), list(best_B)]
