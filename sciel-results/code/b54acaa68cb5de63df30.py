import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    N = 128
    FULL_MASK = (1 << (N + 1)) - 2  # bits 1..128 set

    # Precompute forbidden hyperedges: for each triple (a,b,c) with a<b<c,
    # the triple is forbidden if a+b, a+c, b+c are all in S.
    # For B&B: if a,b,c are all chosen, then a+b, a+c, b+c must be excluded.
    # We precompute for each element x, the list of (other_elements, forced_exclusions)
    # Actually, let's precompute: for each pair (a,b) with a<b and a+b<=N,
    # the set of c values that would create a forbidden triple.
    # If a,b,c are in S and a+b,a+c,b+c are in S, it's forbidden.
    # So if we choose a and b, then for any c where c>a,b and a+c<=N and b+c<=N,
    # if we also choose c AND a+b AND a+c AND b+c, we have a violation.
    
    # Better approach for B&B: 
    # State: chosen set C, forbidden set F, available set A
    # When we add x to C:
    #   For each pair (x, y) where y is in C or will be in C:
    #     For each z where the triple (x,y,z) would be forbidden:
    #       We need to exclude at least one of {x,y,z,a+b,a+c,b+c}
    #     This is complex. Let's use a simpler propagation.
    
    # Simpler: precompute for each element x, the set of elements that conflict with x
    # in the sense that if both are in S, it creates constraints.
    # Actually, the cleanest approach: 
    # For each triple (a,b,c) with a<b<c and a+b<=N and a+c<=N and b+c<=N:
    #   The 6-element set {a,b,c,a+b,a+c,b+c} cannot have all 6 in S.
    #   Equivalently, at least one must be excluded.
    
    # For B&B, we'll use: when adding element x to chosen set,
    # check all triples involving x and two other chosen elements.
    # If such a triple exists where all 6 are available/chosen, we have a conflict.
    
    # Precompute: for each pair (a,b) with a<b, the list of c values (c>b) such that
    # a+c<=N and b+c<=N. These are the "completing" elements for the triple.
    pair_completers = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            completers = []
            max_c = min(N - a, N - b)
            for c in range(b + 1, max_c + 1):
                completers.append(c)
            if completers:
                pair_completers[(a, b)] = completers
    
    # For fast checking: given a mask of chosen elements, check if valid.
    # A triple (a,b,c) is bad if a,b,c,a+b,a+c,b+c are all in mask.
    # We'll precompute for each pair (a,b) the bitmask of completers.
    pair_completer_masks = {}
    for (a, b), comps in pair_completers.items():
        m = 0
        for c in comps:
            m |= (1 << c)
        pair_completer_masks[(a, b)] = m

    def is_valid_mask(mask):
        """Check if no forbidden triple exists in mask."""
        for (a, b), c_mask in pair_completer_masks.items():
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> (a + b)) & 1):
                if (mask & c_mask) != 0:
                    return False
        return True

    # Branch and bound for maximum independent set
    # We model: vertices = 1..128, edges between vertices that cannot both be in S
    # But the constraint is 6-uniform, not pairwise. So we need a different approach.
    
    # Alternative: use the fact that if we fix which elements are in S,
    # we can check validity. For B&B, we'll use a recursive approach:
    # - Maintain a bitmask of available elements
    # - At each step, pick an element, decide to include or exclude it
    # - When including x, check if it creates any forbidden triple with already-included elements
    # - Prune if current_size + popcount(available) <= best_size
    
    import numpy as np
    
    # Precompute popcount for speed
    # We'll use Python's int.bit_count() which is fast in CPython 3.10+
    
    best_size = 0
    best_mask = 0
    node_count = 0
    deadline_reached = False
    
    # Initial greedy solution for a good lower bound
    def greedy_initial():
        mask = 0
        # Try different orderings
        for start in [N // 2 + 1, 1]:
            trial_mask = 0
            if start == N // 2 + 1:
                order = list(range(N, 0, -1))
            else:
                order = list(range(1, N + 1))
            
            for x in order:
                trial = trial_mask | (1 << x)
                if is_valid_mask(trial):
                    trial_mask = trial
            
            if trial_mask.bit_count() > mask.bit_count():
                mask = trial_mask
        return mask

    # Get a good initial solution
    init_mask = greedy_initial()
    best_size = init_mask.bit_count()
    best_mask = init_mask
    
    # Also try a few random restarts for better initial bound
    random.seed(42)
    for _ in range(20):
        trial_mask = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            trial = trial_mask | (1 << x)
            if is_valid_mask(trial):
                trial_mask = trial
        if trial_mask.bit_count() > best_size:
            best_size = trial_mask.bit_count()
            best_mask = trial_mask
    
    # B&B search
    def branch_and_bound(available_mask, chosen_mask, chosen_count):
        nonlocal best_size, best_mask, node_count, deadline_reached
        
        if time.time() > deadline - 2:
            deadline_reached = True
            return
        
        node_count += 1
        
        # Prune: even if we add all available, can't beat best
        if chosen_count + available_mask.bit_count() <= best_size:
            return
        
        # If no more available, update best
        if available_mask == 0:
            if chosen_count > best_size:
                best_size = chosen_count
                best_mask = chosen_mask
            return
        
        # Pick the next element to consider
        # Use a heuristic: pick the element that appears in the most constraints
        # For simplicity, pick the lowest set bit
        x = (available_mask & -available_mask).bit_length() - 1
        remaining = available_mask & ~(1 << x)
        
        # Option 1: Include x
        new_chosen = chosen_mask | (1 << x)
        new_count = chosen_count + 1
        
        # Check if including x is valid
        if is_valid_mask(new_chosen):
            # When we include x, we need to check if it creates forbidden triples
            # with pairs already in chosen_mask. If so, we must exclude the completers.
            # But actually, is_valid_mask already checks the full mask.
            # For propagation: if including x creates a potential forbidden triple,
            # we should remove the conflicting elements from available.
            
            # Propagation: for each pair (a,b) in chosen_mask where a+b is also in chosen_mask,
            # if x is a completer for (a,b), then we'd have a forbidden triple.
            # So we need to either not include x, or exclude a, b, or a+b.
            # Since a,b,a+b are already chosen, we must exclude x.
            # Wait, that's not right. The triple is (a,b,c) where c is the completer.
            # If x is the completer c, and a,b,a+b are in chosen, then including x creates
            # a forbidden triple ONLY IF a+x and b+x are also in chosen or will be.
            # Actually, the constraint is: a,b,c,a+b,a+c,b+c all in S.
            # So if a,b,a+b are in chosen, and x=c, we need a+x and b+x to also be in S.
            # They might not be in chosen yet. So including x doesn't immediately violate.
            # The violation only occurs when ALL SIX are in S.
            
            # So for propagation, we can't easily prune. Let's just check validity.
            branch_and_bound(remaining, new_chosen, new_count)
        
        # Option 2: Exclude x
        branch_and_bound(remaining, chosen_mask, chosen_count)
    
    # Start B&B from the initial solution's complement
    # Available: all elements not in best_mask
    available = FULL_MASK & ~best_mask
    branch_and_bound(available, best_mask, best_size)
    
    # Local search to improve
    if not deadline_reached and time.time() < deadline - 5:
        current_mask = best_mask
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try adding each element
            for x in range(1, N + 1):
                if not (current_mask >> x) & 1:
                    trial = current_mask | (1 << x)
                    if is_valid_mask(trial):
                        current_mask = trial
                        best_size += 1
                        best_mask = current_mask
                        improved = True
                        break
            # If no single addition, try swaps
            if not improved:
                for x in range(1, N + 1):
                    if (current_mask >> x) & 1:
                        base = current_mask & ~(1 << x)
                        for y in range(1, N + 1):
                            if y != x and not (base >> y) & 1:
                                trial = base | (1 << y)
                                if is_valid_mask(trial):
                                    current_mask = trial
                                    best_mask = current_mask
                                    improved = True
                                    break
                        if improved:
                            break
    
    # Final polish
    if time.time() < deadline - 1:
        remaining = FULL_MASK & ~best_mask
        for x in range(1, N + 1):
            if (remaining >> x) & 1:
                trial = best_mask | (1 << x)
                if is_valid_mask(trial):
                    best_mask = trial
    
    # Convert to list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> x) & 1:
            result.append(int(x))
    
    return result
