import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s wall-clock
    
    n = 256
    
    # Precompute conflict neighbors for fast validity checking
    # For each x, we need to know if adding x creates a 3-AP.
    # A 3-AP is (a, b, c) with a+c=2b.
    # If we add x, it forms an AP if:
    # 1. x is the middle: exists y, z in S such that y+z=2x.
    # 2. x is an end: exists y in S such that 2y-x in S (x is larger end) or x-2y in S (x is smaller end).
    
    # To make repair and moves fast, we maintain a set S and for each x, 
    # we can check conflicts by iterating over potential partners.
    # However, for SA, we need to evaluate the delta quickly.
    
    # Let's use a bitset for S for O(1) membership.
    # But for conflict checking, we need to find if any AP exists.
    
    # Strategy:
    # 1. Start with greedy solution.
    # 2. SA Loop:
    #    a. Pick a random move: Add random x not in S, or Remove random x in S.
    #    b. If Add:
    #       - Tentatively add x.
    #       - Check for conflicts. If conflicts exist, we have an invalid state.
    #       - Instead of rejecting, we can "repair" by removing one element from each conflicting AP.
    #       - To simplify and ensure validity at all times (hard constraint), 
    #         we only accept a move if the resulting set is valid.
    #         This is standard "accept if valid" SA.
    #         To improve acceptance rate, we can use "swap" moves: remove y, add x.
    
    # Let's stick to "Flip" moves: 
    #   - If x in S, remove it. (Always valid)
    #   - If x not in S, add it ONLY IF valid. (If invalid, reject move)
    # This might be too restrictive. 
    # Better: "Repair" based SA.
    #   - Propose: Add x (even if invalid).
    #   - Repair: While S is invalid, remove an element involved in an AP. 
    #     Which one? Remove the one with smallest 1/x (largest index) to preserve weight? 
    #     Or remove x itself if it caused the most trouble?
    #     Actually, if we add x and it creates APs, we must remove at least one element from each AP.
    #     A simple heuristic: If adding x creates k APs, we can remove x (backtrack) or remove other elements.
    #     Removing other elements might yield a better set if x has high weight (small index).
    
    # Let's implement a "Greedy Repair" within the SA step.
    # Move: Add x.
    # If valid: Accept.
    # If invalid: 
    #    Identify all APs created by x.
    #    For each AP, we must remove one of the 3 elements.
    #    We can try to remove the "least valuable" element from the APs.
    #    If we remove x, we revert.
    #    If we remove others, we might end up with a valid set with higher sum.
    
    # To keep it simple and fast, let's use a "Local Search" with "Add/Remove" moves 
    # but with a "Conflict Resolution" step that makes it a "Constraint-Handling SA".
    
    # Precompute for each x, the list of pairs (y, z) such that {x,y,z} is a 3-AP.
    # If we add x, we check if any (y,z) are both in S.
    
    conflict_pairs = {i: [] for i in range(1, n + 1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n:
                # AP is (a, b, c)
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
                
    inv = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    def bit(x):
        return 1 << (x - 1)
        
    def get_elements(M):
        return [i for i in range(1, n + 1) if M & bit(i)]
        
    def sum_recip(M):
        s = 0.0
        for i in range(1, n + 1):
            if M & bit(i):
                s += inv[i]
        return s
        
    def is_valid(M):
        # Check if M contains any 3-AP
        # For each x in M, check if any (y,z) in conflict_pairs[x] are both in M
        # This is O(|S| * deg(x)). 
        # To speed up, we can just check the APs formed by the most recent addition?
        # No, for a general check, we need to scan.
        # But in SA, we only add one element. So we only need to check APs involving that element.
        pass
        
    def check_add(x, M):
        # Returns True if adding x to M keeps it valid.
        # Only need to check APs where x is a member.
        for y, z in conflict_pairs[x]:
            if (M & bit(y)) and (M & bit(z)):
                return False
        return True
        
    def find_conflicts(x, M):
        # Returns list of APs (y, z) such that {x, y, z} is an AP and y, z in M.
        confs = []
        for y, z in conflict_pairs[x]:
            if (M & bit(y)) and (M & bit(z)):
                confs.append((y, z))
        return confs
        
    # Initial Greedy Solution
    M_best = 0
    for x in range(1, n + 1):
        if check_add(x, M_best):
            M_best |= bit(x)
            
    score_best = sum_recip(M_best)
    best_mask = M_best
    
    # Simulated Annealing
    # Temperature schedule
    T_start = 0.1  # Scale of 1/x is around 1.0 down to 0.004
    T_end = 0.001
    cooling_rate = 0.999
    
    current_M = M_best
    current_score = score_best
    
    T = T_start
    
    # We will run SA for a fixed number of steps or until time runs out.
    # Each step:
    # 1. Pick a random x in [1, n].
    # 2. If x in current_M:
    #      Remove x. New state is always valid.
    #      Delta = -inv[x].
    #      Accept if random < exp(Delta/T) (always true since Delta<0, wait. 
    #      SA accepts bad moves with prob exp(Delta/T). Delta is negative. 
    #      So we might accept removals. But removing decreases score. 
    #      We only want to remove if it allows adding something better later? 
    #      Standard SA explores the space. 
    # 3. If x not in current_M:
    #      Check if valid to add.
    #      If valid:
    #          Add x. Delta = +inv[x].
    #          Always accept (improvement).
    #      If invalid:
    #          We have a choice: Reject, or "Repair".
    #          Repair: Identify conflicts. 
    #          Let's try a "Swap" repair: 
    #          If adding x creates conflicts, we can try to remove one conflicting element y 
    #          and add x. 
    #          This is a 2-move. 
    #          Let's stick to simple moves first to ensure speed.
    #          If invalid, reject the move.
    
    # To make it more powerful, let's allow "Swap" moves directly.
    # Move type 1: Add x (if valid)
    # Move type 2: Remove x
    # Move type 3: Swap x (in S) with y (not in S) if result is valid.
    
    # Let's use a multi-start approach with a single strong SA run.
    
    random.seed(42)
    
    steps = 0
    while time.time() < deadline:
        if steps % 1000 == 0:
            # Cool down
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        steps += 1
        
        # Choose a move
        r = random.random()
        
        if r < 0.5:
            # Try to Add
            x = random.randint(1, n)
            if not (current_M & bit(x)):
                if check_add(x, current_M):
                    # Valid addition
                    new_M = current_M | bit(x)
                    new_score = current_score + inv[x]
                    # Always accept improvement
                    current_M = new_M
                    current_score = new_score
                    if current_score > score_best:
                        score_best = current_score
                        best_mask = current_M
                else:
                    # Invalid addition. 
                    # Try to repair by swapping?
                    # For now, reject. 
                    pass
        elif r < 0.7:
            # Try to Remove
            # Pick a random x in current_M
            # We need a list of elements in current_M for fast random selection.
            # Maintaining a list is expensive to update.
            # We can just pick random x in [1,n] and check if in M.
            # This is inefficient if M is small.
            # Let's maintain a list of elements in M.
            # Actually, M is likely large (greedy is ~100-150 elements).
            # Picking random x in [1,n] has O(1) rejection overhead if density is high.
            
            x = random.randint(1, n)
            if current_M & bit(x):
                # Remove x
                new_M = current_M & ~bit(x)
                new_score = current_score - inv[x]
                # Accept with probability exp(Delta/T)
                delta = new_score - current_score
                if random.random() < pow(2.718281828, delta / T):
                    current_M = new_M
                    current_score = new_score
        else:
            # Try Swap
            # Pick x in M, y not in M.
            # Try x = random in M, y = random not in M.
            # To pick x in M efficiently, we can sample from [1,n] until we find one in M.
            x = random.randint(1, n)
            while not (current_M & bit(x)):
                x = random.randint(1, n)
                # Safety break if M is empty (shouldn't happen)
                if x == 0: break
                
            y = random.randint(1, n)
            while (current_M & bit(y)):
                y = random.randint(1, n)
                if y == 0: break
            
            if x == 0 or y == 0:
                continue
                
            # Check if swap is valid:
            # Remove x, add y.
            # New set S' = (S \ {x}) U {y}
            # Check if y can be added to S \ {x}.
            temp_M = current_M & ~bit(x)
            if check_add(y, temp_M):
                new_M = temp_M | bit(y)
                new_score = current_score - inv[x] + inv[y]
                delta = new_score - current_score
                # Accept if improvement or with prob exp(delta/T)
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_M = new_M
                    current_score = new_score
                    if current_score > score_best:
                        score_best = current_score
                        best_mask = current_M

    # Final verification
    final_elements = get_elements(best_mask)
    elem_set = set(final_elements)
    # Verify no 3-AP
    valid = True
    for a in final_elements:
        for b in final_elements:
            if a == b:
                continue
            c = 2 * b - a
            if c in elem_set and c != a and c != b:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to greedy if SA corrupted state (shouldn't happen with check_add)
        M_greedy = 0
        for x in range(1, n + 1):
            if check_add(x, M_greedy):
                M_greedy |= bit(x)
        final_elements = get_elements(M_greedy)
        
    if not final_elements:
        final_elements = [1]
        
    return final_elements
