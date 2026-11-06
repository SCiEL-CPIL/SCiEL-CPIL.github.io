import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 511
    # Weights
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
        
    # Conflict Masks
    # conflict_mask[a] has bit b set if gcd(a, b) > 1
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if math.gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Beam Search
    # State: (score, bits)
    # Initialize
    beam = [(0.0, 0)]
    K = 5000  # Beam width
    
    # Process in order of decreasing weight (511 down to 1)
    for a in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        new_beam = []
        w_a = weights[a]
        mask_a = conflict_mask[a]
        
        for score, bits in beam:
            # Option 1: Exclude a
            new_beam.append((score, bits))
            
            # Option 2: Include a (if compatible)
            if (bits & mask_a) == 0:
                new_score = score + w_a
                new_bits = bits | (1 << a)
                new_beam.append((new_score, new_bits))
                
        # Deduplicate and Prune
        # Using a dict to keep max score for each bitset (though score is unique to bitset)
        # Actually, just sort and take top K is simpler and fast enough for K=5000
        new_beam.sort(key=lambda x: x[0], reverse=True)
        
        # Deduplicate: if same bits appear, keep the first one (highest score)
        seen = set()
        pruned_beam = []
        for s, b in new_beam:
            if b not in seen:
                seen.add(b)
                pruned_beam.append((s, b))
                if len(pruned_beam) >= K:
                    break
        beam = pruned_beam
        
    # Return the best solution
    best_score, best_bits = max(beam, key=lambda x: x[0])
    
    # Reconstruct set from bits
    chosen = []
    for a in range(1, N + 1):
        if best_bits & (1 << a):
            chosen.append(a)
            
    return chosen
