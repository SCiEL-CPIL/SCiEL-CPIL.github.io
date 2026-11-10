import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    N = 180
    full_set = set(range(1, N+1))
    
    # 1. Precompute all harmonic triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c => c = ab/(b-a)
    triples = []
    # For speed, we can also precompute neighbors N(y)
    # N(y) is the set of all z such that y, z, and some w form a triple.
    # Actually, we need to know, for a given y, which pairs (u,v) in S block it.
    # It's more efficient to store the triples themselves.
    
    triple_dict = defaultdict(list) # y -> list of (u, v) such that {y,u,v} is a triple
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            num = a * b
            den = b - a
            if den > 0 and num % den == 0:
                c = num // den
                if c > b and c <= N:
                    triples.append((a, b, c))
                    # Add to dict for each element
                    triple_dict[a].append((b, c))
                    triple_dict[b].append((a, c))
                    triple_dict[c].append((a, b))
                    
    # 2. Greedy Initialization
    # Order elements by degree (number of triples they are in)
    # Elements with fewer conflicts are "safer" to add early?
    # Or elements with more conflicts should be decided first?
    # Usually, adding "safe" elements first is good.
    # Let's sort by degree ascending.
    degrees = {i: len(triple_dict[i]) for i in range(1, N+1)}
    sorted_elements = sorted(range(1, N+1), key=lambda x: degrees[x])
    
    current_set = set()
    
    def is_valid_addition(elem, S):
        """Check if adding elem to S creates a forbidden triple."""
        # Check all triples involving elem
        for u, v in triple_dict[elem]:
            if u in S and v in S:
                return False
        return True

    for elem in sorted_elements:
        if time.time() > deadline - 10:
            break
        if is_valid_addition(elem, current_set):
            current_set.add(elem)
            
    # 3. Local Search
    # Strategy: Remove one element, then greedily add as many as possible.
    # Score: Size of the set.
    
    def greedy_fill(S):
        """Add elements to S greedily if possible."""
        # We can iterate through all elements not in S
        # To be fast, we can just iterate 1..N
        for y in range(1, N+1):
            if y not in S:
                if is_valid_addition(y, S):
                    S.add(y)
        return S

    # Initial greedy fill
    current_set = greedy_fill(current_set)
    best_size = len(current_set)
    
    # Local Search Loop
    while time.time() < deadline - 5:
        improved = False
        
        # Calculate for each x in current_set, how many elements y outside would be unblocked
        # if x were removed.
        # y is unblocked by removing x if:
        # 1. y is not in S
        # 2. The only elements in S that block y are x.
        
        # To do this efficiently:
        # For each y not in S, find the set of blockers B_y = { x in S | exists v in S such that {y,x,v} is a triple }
        # Actually, y is blocked if there is ANY triple {y, u, v} with u,v in S.
        # So B_y is the set of all u in S such that there exists v in S with {y,u,v} triple.
        # If |B_y| == 1, let B_y = {x}, then removing x unblocks y.
        
        # We need to count, for each x in S, how many y have B_y = {x}.
        # Let's call this gain[x].
        
        gain = {x: 0 for x in current_set}
        
        # Iterate over y not in S
        for y in range(1, N+1):
            if y in current_set:
                continue
            
            # Find blockers for y
            # Blockers are elements u in S such that there is a triple (y, u, v) with v in S.
            # Note: The triple is {y, u, v}. u and v are the other two.
            # triple_dict[y] contains pairs (u, v).
            # For each (u, v) in triple_dict[y]:
            #   if u in S and v in S: y is blocked.
            #   Specifically, u is a blocker, v is a blocker.
            
            blockers = set()
            for u, v in triple_dict[y]:
                if u in current_set:
                    blockers.add(u)
                if v in current_set:
                    blockers.add(v)
            
            if len(blockers) == 1:
                # The single blocker is the only reason y can't be added
                x = next(iter(blockers))
                gain[x] += 1
            # If len(blockers) > 1, removing one blocker doesn't unblock y immediately
            # (unless removing one also removes the other? No, we remove one x).
            # If len(blockers) == 0, y should have been added in greedy_fill.
            
        # Find x with max gain
        if not gain:
            break
            
        max_gain = max(gain.values())
        if max_gain == 0:
            # No single removal unblocks any new element.
            # Try random removals or break?
            # If max_gain is 0, we are in a local optimum for this move.
            # We can try a random removal to escape?
            # Or break.
            # Let's try a few random removals if time permits.
            import random
            if random.random() < 0.1: # 10% chance to try random move
                 x_to_remove = random.choice(list(current_set))
                 # Test it
                 old_set = set(current_set)
                 current_set.remove(x_to_remove)
                 current_set = greedy_fill(current_set)
                 if len(current_set) > len(old_set):
                     best_size = len(current_set)
                     improved = True
                 else:
                     current_set = old_set
            else:
                break # Stuck
            
        # Select x to remove. If multiple have same max gain, pick one randomly?
        # Or just the first one.
        candidates = [x for x, g in gain.items() if g == max_gain]
        x_to_remove = candidates[0]
        
        # Perform move
        old_set = set(current_set)
        current_set.remove(x_to_remove)
        
        # Greedy fill
        current_set = greedy_fill(current_set)
        
        if len(current_set) > len(old_set):
            best_size = len(current_set)
            improved = True
        else:
            current_set = old_set
            
        if not improved and max_gain == 0:
            # If we tried and didn't improve, and gain was 0, we might be stuck.
            # But if gain > 0 and we didn't improve, it means the "gain" was overestimated
            # (e.g. unblocking y1 blocked y2).
            # In that case, we should try other x's.
            # For simplicity, if we don't improve, we can mark this x as "tried" for this round?
            # Or just break?
            # Let's add a mechanism to try other candidates if the first one fails.
            # Actually, the loop above picks ONE x.
            # Let's modify to try ALL candidates with max gain?
            # Or just break if no improvement after trying the best one.
            # Given N=180, trying a few is cheap.
            
            # Let's just break for now to keep it simple and fast.
            # The greedy fill is strong.
            break

    return sorted(current_set)
