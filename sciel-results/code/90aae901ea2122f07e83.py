import numpy as np
import time
import heapq
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesizes an XOR circuit for the given matrix.
    
    Strategy: 
    1. Convert rows to bitmasks.
    2. Use a Prim's-like algorithm to build a tree of wires (linear combinations).
    3. The "cost" of adding a wire is 1 gate. We prioritize wires that are 
       "useful" for multiple remaining targets (Steiner tree approximation).
    4. Specifically, we look for wires w such that w is a submask of targets 
       or w ^ target is a simple wire (input or 0).
    5. Fallback to direct construction if time runs out.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert rows to integer bitmasks
    target_masks = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Unique targets
    unique_targets = list(set(target_masks))
    
    # Initial state:
    # Available wires: inputs (1<<i) and 0.
    # Wire indices:
    # 0..n-1: inputs
    # n: 0
    # n+1...: generated gates
    
    available_masks = set()
    for i in range(n):
        available_masks.add(1 << i)
    available_masks.add(0)
    
    mask_to_wire = {}
    for i in range(n):
        mask_to_wire[1 << i] = i
    mask_to_wire[0] = n
    
    next_wire = n + 1
    gates = []
    
    # Targets not yet solved
    remaining_targets = set(unique_targets)
    
    # Remove targets that are already available (inputs or 0)
    for t in list(remaining_targets):
        if t in available_masks:
            remaining_targets.remove(t)
    
    # Prim's-like growth:
    # We maintain a set of "frontier" candidates.
    # At each step, we pick the candidate wire w that maximizes "utility".
    # Utility is defined as the number of remaining targets T such that 
    # T can be completed in 1 step using w (i.e., T ^ w is in available_masks).
    # Additionally, we give a bonus if w itself is a target.
    
    # To make this efficient, we don't just look at pairs of *all* available wires,
    # but we can be smarter. However, for n<=16, the number of available wires 
    # stays manageable if we prune.
    
    # Optimization: 
    # Instead of checking all pairs O(|A|^2) every step, we can maintain a 
    # priority queue of candidate wires based on their potential.
    # But since the "utility" depends on the *current* set of available wires 
    # and remaining targets, the scores change dynamically.
    
    # Alternative: "BFS with Pruning"
    # Let's stick to the iterative greedy but with a better scoring function 
    # that explicitly rewards sharing.
    
    # Scoring function for a candidate wire w (formed by a ^ b):
    # 1. If w is in remaining_targets: +1000 (direct solve)
    # 2. For each T in remaining_targets:
    #    If (T ^ w) is in available_masks: +10 (T can be solved in next step)
    #    If (T ^ w) is in remaining_targets: +1 (w helps reduce T to another target? No, that's circular)
    #    Actually, if T ^ w is a "simple" wire (popcount <= 2), it's cheap to build.
    
    # Let's implement the loop.
    
    while remaining_targets:
        if time.time() > DEADLINE:
            break
        
        avail_list = list(available_masks)
        # If the list is huge, sampling might be needed, but for now assume it's small enough.
        # For 16x16, we might generate up to ~100-200 wires. 200^2 = 40k pairs. 40k * 16 targets = 640k ops.
        # This is fast in Python if optimized, but let's be careful.
        
        best_w = None
        best_score = -1
        best_a = None
        best_b = None
        
        # We only need to consider pairs where a < b to avoid duplicates
        # and where a ^ b is NOT in available_masks.
        
        # Optimization: Pre-compute a set of available masks for O(1) lookup.
        # We already have available_masks.
        
        # To speed up the inner loop, we can pre-compute the "complements" of targets.
        # For a candidate w, score = sum(1 for T in remaining if (T^w) in available)
        
        # Let's iterate pairs.
        # If len(avail_list) is large, this becomes slow.
        # Heuristic: If len(avail_list) > 500, we might need to be more selective.
        # But let's try the full search first with a time check inside.
        
        L = len(avail_list)
        # If L is very large, we might skip to a simpler heuristic
        if L > 1000:
             # Fallback: Pick a random pair or a pair that reduces popcount of a target
             # For now, let's just pick the first promising pair found
             for i in range(L):
                 a = avail_list[i]
                 for j in range(i+1, L):
                     b = avail_list[j]
                     w = a ^ b
                     if w in available_masks:
                         continue
                     # Quick check
                     if w in remaining_targets:
                         best_w = w
                         best_a = a
                         best_b = b
                         best_score = 100000
                         break
                 if best_w is not None:
                     break
             if best_w is None:
                 # Just pick any new wire
                 for i in range(L):
                     a = avail_list[i]
                     for j in range(i+1, L):
                         b = avail_list[j]
                         w = a ^ b
                         if w not in available_masks:
                             best_w = w
                             best_a = a
                             best_b = b
                             break
                     if best_w is not None:
                         break
        
        else:
            # Full search for best utility
            # To optimize, we can group targets by their "distance" to available wires?
            # No, just compute directly.
            
            for i in range(L):
                a = avail_list[i]
                for j in range(i + 1, L):
                    b = avail_list[j]
                    w = a ^ b
                    
                    if w in available_masks:
                        continue
                    
                    score = 0
                    # Direct hit
                    if w in remaining_targets:
                        score += 100000
                    
                    # Check one-step completions
                    # T is solvable if T^w is in available_masks
                    # We can speed this up by checking if (T^w) is in available_masks
                    for t in remaining_targets:
                        if (t ^ w) in available_masks:
                            score += 100
                            # Early exit if score is already huge?
                            if score > 10000:
                                break
                    
                    if score > best_score:
                        best_score = score
                        best_w = w
                        best_a = a
                        best_b = b
                        
                        # If we found a direct hit, we can stop early
                        if best_w in remaining_targets:
                            break
                if best_w in remaining_targets:
                    break
        
        if best_w is None:
            # No new wire found? Should not happen unless we've exhausted the space
            # which is impossible for n<=16 and small target count.
            break
        
        # Add the gate
        wire_a = mask_to_wire[best_a]
        wire_b = mask_to_wire[best_b]
        
        gates.append([wire_a, wire_b])
        new_wire_idx = next_wire
        next_wire += 1
        
        mask_to_wire[best_w] = new_wire_idx
        available_masks.add(best_w)
        
        # Update remaining targets
        # If best_w is a target, remove it.
        # Also, check if any other target is now solvable? 
        # No, "solvable" means the wire exists. 
        # The target is "satisfied" when its mask is in available_masks.
        # So we just remove best_w if it's in remaining_targets.
        # Wait, what if T ^ w was in available, but T itself wasn't?
        # T is not in available yet. We need another gate to create T.
        # The loop will handle that in the next iteration.
        
        if best_w in remaining_targets:
            remaining_targets.remove(best_w)
        
        # Optimization: Check if any remaining target is now in available_masks?
        # This can only happen if best_w == target. 
        # Because we only add best_w to available_masks.
        # So the above check is sufficient.
        
    # Fallback for unsolved targets
    unsolved = set(target_masks) - available_masks
    for t in unsolved:
        if time.time() > DEADLINE:
            # Critical: Must return *something* valid.
            # If we can't solve it, we might have to output 0? 
            # No, that's incorrect. 
            # We must construct it, even if expensive.
            # But if we are out of time, we can't even construct it safely?
            # The contract says "invalid result... scores 0".
            # So we MUST finish construction.
            # Let's force a simple chain construction which is O(n) per target.
            pass
        
        # Simple chain construction from inputs
        # Find best available wire to reduce popcount
        best_w = 0
        min_pop = t.bit_count()
        # Limit search time here
        search_start = time.time()
        for w in available_masks:
            if time.time() - search_start > 1.0:
                break
            diff = t ^ w
            p = diff.bit_count()
            if p < min_pop:
                min_pop = p
                best_w = w
        
        rem = t ^ best_w
        
        if rem in available_masks:
            wire_bw = mask_to_wire[best_w]
            wire_rem = mask_to_wire[rem]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            available_masks.add(t)
        else:
            # Build rem from inputs
            bits = [i for i in range(n) if (rem >> i) & 1]
            
            if len(bits) == 0:
                wire_rem = n
            elif len(bits) == 1:
                wire_rem = bits[0]
            else:
                curr_wire = bits[0]
                for b in bits[1:]:
                    gates.append([curr_wire, b])
                    new_wire_idx = next_wire
                    next_wire += 1
                    curr_wire = new_wire_idx
                wire_rem = curr_wire
            
            wire_bw = mask_to_wire[best_w]
            gates.append([wire_bw, wire_rem])
            new_wire_idx = next_wire
            next_wire += 1
            mask_to_wire[t] = new_wire_idx
            available_masks.add(t)
            
    # Map outputs
    outputs = []
    for t in target_masks:
        if t in mask_to_wire:
            outputs.append(mask_to_wire[t])
        else:
            # Should not happen
            outputs.append(n)
            
    return {'gates': gates, 'outputs': outputs}
