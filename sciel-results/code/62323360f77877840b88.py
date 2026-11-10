import time
import random
from itertools import combinations
import sys

# Precompute K4s and their edge indices
# Edge index mapping: (i, j) -> 15*i - i*(i-1)//2 + (j-i-1)
def _edge_index(i, j):
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Build list of K4s
_K4S = []
_K4_EDGE_INDICES = []
for quad in combinations(range(16), 4):
    edges = []
    for a, b in combinations(quad, 2):
        edges.append(_edge_index(a, b))
    _K4S.append(quad)
    _K4_EDGE_INDICES.append(tuple(edges))

_N_K4S = len(_K4S)

# Map edge index to list of K4 indices it belongs to
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, edges in enumerate(_K4_EDGE_INDICES):
    for e in edges:
        _EDGE_TO_K4S[e].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # State variables
    # colours: list of 120 ints
    # k4_colour_counts: list of 1820 dicts {colour: count}
    # violations: int count of K4s with < 5 distinct colours
    
    def init_state(K):
        colours = [random.randint(0, K - 1) for _ in range(120)]
        k4_colour_counts = [{} for _ in range(_N_K4S)]
        violations = 0
        
        for k4_idx, edges in enumerate(_K4_EDGE_INDICES):
            counts = {}
            for e in edges:
                c = colours[e]
                counts[c] = counts.get(c, 0) + 1
            k4_colour_counts[k4_idx] = counts
            if len(counts) < 5:
                violations += 1
        return colours, k4_colour_counts, violations

    def try_change(colours, k4_colour_counts, violations, edge_idx, new_colour):
        old_colour = colours[edge_idx]
        if old_colour == new_colour:
            return colours, k4_colour_counts, violations
        
        new_violations = violations
        affected_k4s = _EDGE_TO_K4S[edge_idx]
        
        # Check if the change is valid for all affected K4s
        # We need to see if the number of distinct colours drops below 5
        # or if it was below 5 and becomes >= 5
        
        # To be safe and simple, we calculate the new state for affected K4s
        # and check if any new violation is created or if we can accept it.
        
        # For a simple hill climb, we only accept if violations decrease or stay same.
        # Let's check if the move is valid (doesn't increase violations)
        
        # Optimization: Pre-check
        # For each affected K4:
        #   old_counts = k4_colour_counts[k4_idx]
        #   new_counts = old_counts copy
        #   new_counts[old_colour] -= 1
        #   if new_counts[old_colour] == 0: del new_counts[old_colour]
        #   new_counts[new_colour] = new_counts.get(new_colour, 0) + 1
        #   new_distinct = len(new_counts)
        #   old_distinct = len(old_counts)
        
        #   if old_distinct < 5 and new_distinct >= 5: violation resolved (-1)
        #   if old_distinct >= 5 and new_distinct < 5: violation created (+1)
        #   else: no change
        
        # We want to minimize violations.
        
        # Let's just simulate the change and count delta violations.
        delta_violations = 0
        valid_move = True
        
        # We'll store the new counts if we accept
        new_counts_cache = {}
        
        for k4_idx in affected_k4s:
            old_counts = k4_colour_counts[k4_idx]
            old_distinct = len(old_counts)
            
            # Create a modified count dict
            # We don't want to mutate the original yet
            # We can check conditions without full copy if we are careful, 
            # but dict copy is cheap enough for 1820 K4s * 120 edges * iterations?
            # 120 * 1820 is too big for full copy per step.
            # But each edge is in only 15 K4s (C(15,3) = 455? No. 
            # Edge (u,v) is in K4s formed by {u,v} + 2 other vertices from 14.
            # C(14,2) = 91 K4s per edge.
            # So 91 dict copies per edge move. That's fine.
            
            new_counts = dict(old_counts)
            
            # Remove old colour
            if old_colour in new_counts:
                new_counts[old_colour] -= 1
                if new_counts[old_colour] == 0:
                    del new_counts[old_colour]
            
            # Add new colour
            new_counts[new_colour] = new_counts.get(new_colour, 0) + 1
            
            new_distinct = len(new_counts)
            
            if old_distinct < 5:
                if new_distinct >= 5:
                    delta_violations -= 1
                else:
                    # Still violating, no change in count
                    pass
            else: # old_distinct >= 5
                if new_distinct < 5:
                    delta_violations += 1
                    # If we allow simulated annealing, we might accept this.
                    # For now, strict hill climb: reject if delta > 0
                    valid_move = False 
                    # Actually, we should check ALL affected K4s to get total delta
                    # Let's continue checking to see total delta, then decide.
                    # But if we want strict hill climb, we can break early if delta > 0?
                    # No, because another K4 might be fixed.
                    # Let's accumulate total delta.
            
            new_counts_cache[k4_idx] = new_counts
        
        # If we want to accept moves that increase violations (Simulated Annealing),
        # we would use delta_violations.
        # Here we use strict hill climbing: accept if delta_violations < 0.
        # If delta == 0, accept with some probability to escape local minima?
        # Let's try strict first, then add randomization if stuck.
        
        if delta_violations < 0:
            # Apply change
            colours[edge_idx] = new_colour
            for k4_idx in affected_k4s:
                k4_colour_counts[k4_idx] = new_counts_cache[k4_idx]
            new_violations = violations + delta_violations
            return colours, k4_colour_counts, new_violations, True
        
        return colours, k4_colour_counts, violations, False

    def try_merge(colours, k4_colour_counts, violations, source_colour, target_colour):
        if source_colour == target_colour:
            return colours, k4_colour_counts, violations, False
            
        # Check if merge is valid
        # We need to check all K4s.
        # If a K4 has source_colour, its distinct count might drop.
        # Specifically, if the K4 has source_colour and target_colour, distinct count decreases by 1.
        # If it has source_colour but not target_colour, distinct count stays same (source becomes target).
        # If it has target_colour but not source_colour, no change.
        
        # We only care about K4s that contain at least one edge of source_colour.
        # And we need to ensure no K4 drops below 5 distinct colours.
        
        # Optimization:
        # For each K4, if it contains source_colour:
        #   if it also contains target_colour:
        #       new_distinct = old_distinct - 1
        #       if new_distinct < 5: invalid
        #   else:
        #       new_distinct = old_distinct (source maps to target, which is already there? No.
        #       If source is present and target is NOT present:
        #       The set of colours becomes (S \ {source}) U {target}.
        #       Since target was not in S, and source was in S, the size remains the same.
        #       Wait. If S={1,2,3,4,5} and we map 1->6. New S={2,3,4,5,6}. Size 5.
        #       If S={1,2,3,4} (invalid already) and we map 1->2. New S={2,3,4}. Size 3.
        
        # So the danger is only when source and target are BOTH in the K4.
        # Then distinct count drops by 1.
        
        # So we only need to check K4s that contain BOTH source and target colours.
        
        # Let's identify such K4s.
        # We can iterate through all K4s.
        
        valid = True
        for k4_idx in range(_N_K4S):
            counts = k4_colour_counts[k4_idx]
            if source_colour in counts:
                if target_colour in counts:
                    new_distinct = len(counts) - 1
                    if new_distinct < 5:
                        valid = False
                        break
                # If target not in counts, distinct count remains same.
                # If source not in counts, no change.
        
        if not valid:
            return colours, k4_colour_counts, violations, False
        
        # Apply merge
        new_colours = list(colours)
        for i, c in enumerate(new_colours):
            if c == source_colour:
                new_colours[i] = target_colour
        
        # Update K4 counts
        new_k4_counts = [dict(c) for c in k4_colour_counts] # Shallow copy of dicts? No, we need to update specific ones.
        # Actually, we can just update the ones that changed.
        # But it's easier to rebuild or update in place.
        # Let's update in place for efficiency.
        
        for k4_idx in range(_N_K4S):
            counts = k4_colour_counts[k4_idx]
            if source_colour in counts:
                sc = counts[source_colour]
                del counts[source_colour]
                counts[target_colour] = counts.get(target_colour, 0) + sc
                # Note: if target_colour was already there, we just add to it.
                # If it wasn't, we add it.
                # The logic above for validity ensured that if target was there, distinct count >= 5.
                # If target was NOT there, distinct count was >= 5 (since source was there and valid state implies >=5? 
                # Wait, if state is valid, all K4s have >=5 distinct colours.
                # So if source is there, distinct >= 5.
                # If target is NOT there, new distinct = old distinct. So >= 5.
                # If target IS there, new distinct = old distinct - 1. We checked this is >= 5.
        
        return new_colours, k4_colour_counts, violations, True

    # Main Search Loop
    best_colours = None
    best_K = 17 # Initial upper bound from seed
    
    # We will try to find a valid solution with K colours.
    # Start with K=13. If found, check if we can merge to 12 (impossible per bounds, but good to know).
    # Bounds say 13 <= f(16) <= 15. So we aim for 13.
    
    for K in range(13, 17):
        if time.time() > deadline:
            break
            
        # Multiple restarts for each K
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            colours, k4_counts, violations = init_state(K)
            
            # Local Search
            no_improvement_count = 0
            max_no_improvement = 1000
            
            while time.time() < deadline and violations > 0:
                if no_improvement_count > max_no_improvement:
                    break
                
                # Pick a random edge involved in a violation
                # To do this efficiently, we can maintain a list of violating K4s.
                # For now, pick a random edge and check if it's in a violation.
                # Or pick a random violating K4 and pick a random edge from it.
                
                # Let's pick a random K4 that is violating.
                # We don't have a list of violating K4s. We just have the count.
                # Let's find one.
                viol_k4_idx = -1
                for k4_idx in range(_N_K4S):
                    if len(k4_counts[k4_idx]) < 5:
                        viol_k4_idx = k4_idx
                        break
                
                if viol_k4_idx == -1:
                    violations = 0
                    break
                    
                # Pick a random edge from this violating K4
                edges_in_k4 = _K4_EDGE_INDICES[viol_k4_idx]
                edge_idx = random.choice(edges_in_k4)
                
                # Try all colours for this edge
                best_new_violations = violations
                best_new_colour = colours[edge_idx]
                best_new_counts = None
                best_new_colours = None
                
                for new_c in range(K):
                    if new_c == colours[edge_idx]:
                        continue
                    
                    # Simulate change
                    old_c = colours[edge_idx]
                    delta = 0
                    new_counts_cache = {}
                    valid_for_hillclimb = True # We accept if delta < 0
                    
                    for k4_idx in _EDGE_TO_K4S[edge_idx]:
                        old_counts = k4_counts[k4_idx]
                        old_distinct = len(old_counts)
                        
                        new_counts = dict(old_counts)
                        if old_c in new_counts:
                            new_counts[old_c] -= 1
                            if new_counts[old_c] == 0:
                                del new_counts[old_c]
                        
                        new_counts[new_c] = new_counts.get(new_c, 0) + 1
                        new_distinct = len(new_counts)
                        
                        if old_distinct < 5:
                            if new_distinct >= 5:
                                delta -= 1
                        else:
                            if new_distinct < 5:
                                delta += 1
                        new_counts_cache[k4_idx] = new_counts
                    
                    if delta < 0:
                        if delta < best_new_violations - violations: # We want most negative delta
                            best_new_violations = violations + delta
                            best_new_colour = new_c
                            best_new_counts = new_counts_cache
                            best_new_colours = list(colours)
                            best_new_colours[edge_idx] = new_c
                            
                    # Allow delta == 0 with low probability to escape plateaus?
                    # For now, strict.
                
                if best_new_violations < violations:
                    colours = best_new_colours
                    for k4_idx in _EDGE_TO_K4S[edge_idx]:
                        k4_counts[k4_idx] = best_new_counts[k4_idx]
                    violations = best_new_violations
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    # Perturb: change a random edge to a random colour
                    if no_improvement_count % 10 == 0:
                        pe = random.randint(0, 119)
                        pc = random.randint(0, K - 1)
                        old_pc = colours[pe]
                        colours[pe] = pc
                        # Update counts
                        for k4_idx in _EDGE_TO_K4S[pe]:
                            counts = k4_counts[k4_idx]
                            if old_pc in counts:
                                counts[old_pc] -= 1
                                if counts[old_pc] == 0:
                                    del counts[old_pc]
                            counts[pc] = counts.get(pc, 0) + 1
                            # Recalculate violations for these K4s?
                            # We need to update global violation count.
                            # This is tricky. Let's just recalculate global violations after perturbation.
                            # Or better, just restart if stuck too long.
                        
                        # Recalculate violations
                        violations = 0
                        for k4_idx in range(_N_K4S):
                            if len(k4_counts[k4_idx]) < 5:
                                violations += 1
                        no_improvement_count = 0

            if violations == 0:
                # Valid solution with K colours
                # Try to merge colours to reduce K
                current_K = K
                merged = True
                while merged:
                    merged = False
                    for src in range(current_K - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        for tgt in range(src):
                            new_colours, new_k4_counts, new_violations, ok = try_merge(
                                colours, k4_counts, violations, src, tgt
                            )
                            if ok:
                                colours = new_colours
                                k4_counts = new_k4_counts
                                current_K -= 1
                                merged = True
                                break
                        if merged:
                            break
                            
                if current_K < best_K:
                    best_K = current_K
                    best_colours = list(colours)
                    # If we reached 13, we can stop searching for lower K (since bound is 13)
                    if best_K == 13:
                        # Return immediately? Or try to optimize further?
                        # 13 is the lower bound. So we can return.
                        # But we need to return 120 integers.
                        return best_colours
                # If we found a valid solution, we might want to try to improve it further
                # or just return if it's good enough.
                # Since we are iterating K from 13 up, finding one at 13 is optimal.
                # Finding one at 14 is good, but we might find 13 later?
                # No, we try 13 first. If we fail 13, we try 14.
                # So if we succeed at 14, we should keep searching for 13?
                # The loop structure is: for K in 13..16.
                # If we find a solution at 14, we store it. But we should continue to try 13?
                # No, we try 13 first. If we didn't find 13 in the allocated time/restarts, we move to 14.
                # If we find 14, we store it. Then we try 15, 16.
                # But we want the MINIMUM. So if we have a 14, and we haven't found 13, we should keep trying 13?
                # The current loop does: try 13 (50 restarts). If fail, try 14.
                # This is fine.
                pass
            else:
                # Failed to find valid solution for this K in this restart
                pass

    # If we didn't find 13, return best found (e.g. 14 or 15)
    if best_colours is not None:
        return best_colours
    
    # Fallback: Return the seed solution (17 colours)
    # colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Normalise
    mapping = {}
    next_c = 0
    final_colours = []
    for i in range(16):
        for j in range(i + 1, 16):
            c = (i + j) % 17
            if c not in mapping:
                mapping[c] = next_c
                next_c += 1
            final_colours.append(mapping[c])
    return final_colours

construct()
