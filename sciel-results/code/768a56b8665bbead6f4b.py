"""Rewritten program using exact cardinality-indexed subset-sum DP with greedy construction."""

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Approach: Beam Search with Bitset DP for subset sums.
    """
    
    def try_add(sums, x, max_k):
        # sums is a list of ints (bitsets) of length max_k + 1
        # Returns new_sums list of length max_k + 2 if valid, else None
        new_sums = [0] * (max_k + 2)
        new_sums[0] = 1
        
        for k in range(1, max_k + 2):
            part1 = sums[k] if k <= max_k else 0
            part2 = (sums[k-1] << x)
            new_sums[k] = part1 | part2
            
        # Check validity: disjointness
        for i in range(max_k + 2):
            si = new_sums[i]
            if si == 0:
                continue
            for j in range(i + 1, max_k + 2):
                sj = new_sums[j]
                if sj != 0 and (si & sj):
                    return None
        return new_sums

    beam_width = 20
    max_size = 60
    candidates = list(range(1, 257))
    
    # State: (current_set, sums_bitsets, max_k)
    init_sums = [1] + [0]*max_size
    beam = [( [], init_sums, 0 )]
    
    best_result = []
    
    for step in range(max_size):
        next_beam = []
        
        for current_set, sums, max_k in beam:
            used = set(current_set)
            
            best_score_for_state = float('inf')
            best_next_state = None
            best_x = -1
            
            for x in candidates:
                if x in used:
                    continue
                
                new_sums = try_add(sums, x, max_k)
                if new_sums is None:
                    continue
                
                # Calculate score: Sum of popcounts
                score = 0
                for k in range(max_k + 2):
                    score += new_sums[k].bit_count()
                
                # Tie-breaker: Prefer smaller x
                if score < best_score_for_state or (score == best_score_for_state and x < best_x):
                    best_score_for_state = score
                    best_next_state = new_sums
                    best_x = x
            
            if best_next_state is not None:
                new_set = current_set + [best_x]
                next_beam.append((new_set, best_next_state, max_k + 1))
        
        if not next_beam:
            break
            
        # Sort next_beam by score to keep the most promising states
        def get_score(state):
            _, sums, mk = state
            s = 0
            for k in range(mk + 1):
                s += sums[k].bit_count()
            return s
            
        next_beam.sort(key=get_score)
        
        if len(next_beam) > beam_width:
            next_beam = next_beam[:beam_width]
            
        beam = next_beam
        
        # Update best result
        for state in beam:
            if len(state[0]) > len(best_result):
                best_result = state[0]
                
        if len(best_result) == max_size:
            break
            
    return [int(x) for x in best_result]


# Let's test with a quick sanity check
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Set: {result}")
    
    # Verify the constraint
    if len(result) > 0:
        # Compute all subset sums by cardinality
        sums_by_card = [set() for _ in range(len(result) + 1)]
        sums_by_card[0] = {0}
        for i, x in enumerate(result):
            # Update in reverse to avoid using x multiple times
            for k in range(i + 1, 0, -1):
                for s in sums_by_card[k - 1]:
                    sums_by_card[k].add(s + x)
        
        # Check disjointness
        valid = True
        for k in range(len(result) + 1):
            for j in range(k + 1, len(result) + 1):
                if sums_by_card[k] & sums_by_card[j]:
                    valid = False
                    break
            if not valid:
                break
        
        print(f"Valid: {valid}")
