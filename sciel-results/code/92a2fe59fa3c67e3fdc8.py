"""
Improved program for two Sidon sets with disjoint difference sets.
Uses bitset-accelerated collision checks and randomized local search.
"""

import time
import random
import sys

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # Safety margin before 90s kill

# Bitset sizes: sums range 2..512 (indices 2..512), diffs range -255..255 (offset 255, indices 0..510)
SUM_OFFSET = 2
SUM_BITS = 511  # sums from 2 to 512 inclusive
DIFF_OFFSET = 255
DIFF_BITS = 511  # diffs from -255 to 255 inclusive

class SidonState:
    """Tracks pair sums and nonzero differences for a set using bitsets."""
    def __init__(self):
        self.els = set()
        self.sums_bits = 0   # bitset of sums (a+b for a<=b)
        self.diffs_bits = 0  # bitset of nonzero diffs (a-b for a!=b)
        
    def size(self):
        return len(self.els)
        
    def can_add(self, x, other_diffs_bits=0):
        """
        Check if x can be added to this set.
        other_diffs_bits: bitset of differences from the other set.
        """
        # Check sums: x + a for a in els, and x + x
        # Sum x+a is valid if bit (x+a - SUM_OFFSET) is not set
        for a in self.els:
            s = x + a
            if s < 2 or s > 512:
                continue
            if (self.sums_bits >> (s - SUM_OFFSET)) & 1:
                return False
        sxx = x + x
        if sxx >= 2 and sxx <= 512:
            if (self.sums_bits >> (sxx - SUM_OFFSET)) & 1:
                return False
            
        # Check diffs: x - a and a - x for a in els
        for a in self.els:
            d1 = x - a
            d2 = a - x
            # Check against own diffs
            if -255 <= d1 <= 255:
                if (self.diffs_bits >> (d1 + DIFF_OFFSET)) & 1:
                    return False
            if -255 <= d2 <= 255:
                if (self.diffs_bits >> (d2 + DIFF_OFFSET)) & 1:
                    return False
            # Check against other diffs
            if other_diffs_bits:
                if -255 <= d1 <= 255:
                    if (other_diffs_bits >> (d1 + DIFF_OFFSET)) & 1:
                        return False
                if -255 <= d2 <= 255:
                    if (other_diffs_bits >> (d2 + DIFF_OFFSET)) & 1:
                        return False
        return True

    def add(self, x):
        """Add x to the set. Assumes can_add is true."""
        self.els.add(x)
        for a in self.els:
            if a != x:
                s = a + x
                if 2 <= s <= 512:
                    self.sums_bits |= (1 << (s - SUM_OFFSET))
                d1 = a - x
                d2 = x - a
                if -255 <= d1 <= 255:
                    self.diffs_bits |= (1 << (d1 + DIFF_OFFSET))
                if -255 <= d2 <= 255:
                    self.diffs_bits |= (1 << (d2 + DIFF_OFFSET))
        sxx = x + x
        if 2 <= sxx <= 512:
            self.sums_bits |= (1 << (sxx - SUM_OFFSET))
        
    def remove(self, x):
        """Remove x from the set."""
        if x not in self.els:
            return
        self.els.remove(x)
        # Rebuild sums and diffs bits from scratch (simpler and correct)
        self.sums_bits = 0
        self.diffs_bits = 0
        for a in self.els:
            for b in self.els:
                if a <= b:
                    s = a + b
                    if 2 <= s <= 512:
                        self.sums_bits |= (1 << (s - SUM_OFFSET))
                if a != b:
                    d = a - b
                    if -255 <= d <= 255:
                        self.diffs_bits |= (1 << (d + DIFF_OFFSET))

def greedy_build(state_a, state_b, rng, deadline):
    """Greedily build sets A and B."""
    # Shuffle candidates
    candidates_a = list(range(1, N + 1))
    candidates_b = list(range(1, N + 1))
    rng.shuffle(candidates_a)
    rng.shuffle(candidates_b)
    
    # Build A
    for x in candidates_a:
        if state_a.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            return
        if state_a.can_add(x):
            state_a.add(x)
            
    # Build B, avoiding A's diffs
    for x in candidates_b:
        if state_b.size() >= MAX_SIZE:
            break
        if time.time() > deadline:
            return
        if state_b.can_add(x, state_a.diffs_bits):
            state_b.add(x)

def local_search(state_a, state_b, rng, deadline):
    """
    Local search with 1-opt (add) and 1-for-1 swaps.
    Returns the best state found.
    """
    best_total = state_a.size() + state_b.size()
    best_a_els = set(state_a.els)
    best_b_els = set(state_b.els)
    
    # Pre-compute all possible values for quick iteration
    all_vals = list(range(1, N + 1))
    
    while time.time() < deadline:
        progress = False
        
        # --- Phase 1: Try to add (1-opt) ---
        # Try adding to A
        if state_a.size() < MAX_SIZE:
            # Randomize order to avoid bias
            candidates = [x for x in all_vals if x not in state_a.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_a.can_add(x, state_b.diffs_bits):
                    state_a.add(x)
                    progress = True
                    break
        
        # Try adding to B
        if state_b.size() < MAX_SIZE:
            candidates = [x for x in all_vals if x not in state_b.els]
            rng.shuffle(candidates)
            for x in candidates:
                if state_b.can_add(x, state_a.diffs_bits):
                    state_b.add(x)
                    progress = True
                    break
        
        # --- Phase 2: Try 1-for-1 swaps ---
        # If we didn't progress with adds, try swapping one element for another.
        if not progress:
            # Try swapping in A: remove x from A, add y to A
            if state_a.size() > 1:
                a_candidates = list(state_a.els)
                rng.shuffle(a_candidates)
                for x_remove in a_candidates:
                    # Save state
                    old_a_els = set(state_a.els)
                    old_a_sums_bits = state_a.sums_bits
                    old_a_diffs_bits = state_a.diffs_bits
                    
                    state_a.remove(x_remove)
                    
                    # Try to add one element to A
                    added = False
                    if state_a.size() < MAX_SIZE:
                        cands_a = [x for x in all_vals if x not in state_a.els and x != x_remove]
                        rng.shuffle(cands_a)
                        for x_add in cands_a:
                            if state_a.can_add(x_add, state_b.diffs_bits):
                                state_a.add(x_add)
                                added = True
                                break
                    
                    # If we added, check if total size improved or stayed same
                    # Since we removed 1 and added 1, total is same.
                    # Accept to explore different configurations.
                    if added:
                        progress = True
                        current_total = state_a.size() + state_b.size()
                        if current_total > best_total:
                            best_total = current_total
                            best_a_els = set(state_a.els)
                            best_b_els = set(state_b.els)
                        # Accept the move even if size unchanged (exploration)
                        # But only if it might lead to future improvements
                        # For now, accept with some probability
                        if rng.random() < 0.3:  # 30% chance to accept neutral swap
                            pass  # Keep the change
                        else:
                            # Revert
                            state_a.els = old_a_els
                            state_a.sums_bits = old_a_sums_bits
                            state_a.diffs_bits = old_a_diffs_bits
                            progress = False
                    else:
                        # Revert
                        state_a.els = old_a_els
                        state_a.sums_bits = old_a_sums_bits
                        state_a.diffs_bits = old_a_diffs_bits
                        
                    if progress:
                        break
            
            # Try swapping in B: remove x from B, add y to B
            if not progress and state_b.size() > 1:
                b_candidates = list(state_b.els)
                rng.shuffle(b_candidates)
                for x_remove in b_candidates:
                    old_b_els = set(state_b.els)
                    old_b_sums_bits = state_b.sums_bits
                    old_b_diffs_bits = state_b.diffs_bits
                    
                    state_b.remove(x_remove)
                    
                    added = False
                    if state_b.size() < MAX_SIZE:
                        cands_b = [x for x in all_vals if x not in state_b.els and x != x_remove]
                        rng.shuffle(cands_b)
                        for x_add in cands_b:
                            if state_b.can_add(x_add, state_a.diffs_bits):
                                state_b.add(x_add)
                                added = True
                                break
                    
                    if added:
                        progress = True
                        current_total = state_a.size() + state_b.size()
                        if current_total > best_total:
                            best_total = current_total
                            best_a_els = set(state_a.els)
                            best_b_els = set(state_b.els)
                        if rng.random() < 0.3:
                            pass
                        else:
                            state_b.els = old_b_els
                            state_b.sums_bits = old_b_sums_bits
                            state_b.diffs_bits = old_b_diffs_bits
                            progress = False
                    else:
                        state_b.els = old_b_els
                        state_b.sums_bits = old_b_sums_bits
                        state_b.diffs_bits = old_b_diffs_bits
                        
                    if progress:
                        break
            
            # Try cross-swap: remove from A, add to B (or vice versa)
            if not progress:
                # Remove from A, add to B
                if state_a.size() > 1 and state_b.size() < MAX_SIZE:
                    a_candidates = list(state_a.els)
                    rng.shuffle(a_candidates)
                    for x_remove in a_candidates:
                        old_a_els = set(state_a.els)
                        old_a_sums_bits = state_a.sums_bits
                        old_a_diffs_bits = state_a.diffs_bits
                        old_b_els = set(state_b.els)
                        old_b_sums_bits = state_b.sums_bits
                        old_b_diffs_bits = state_b.diffs_bits
                        
                        state_a.remove(x_remove)
                        
                        added = False
                        cands_b = [x for x in all_vals if x not in state_b.els and x != x_remove]
                        rng.shuffle(cands_b)
                        for x_add in cands_b:
                            if state_b.can_add(x_add, state_a.diffs_bits):
                                state_b.add(x_add)
                                added = True
                                break
                        
                        if added:
                            # Size decreased by 1 (removed from A, added to B, but A lost 1, B gained 1)
                            # Net change: 0 if both had room, but A size decreased
                            # Actually: A lost 1, B gained 1. Total same.
                            # But this might free up diffs in A to allow more adds later.
                            progress = True
                            current_total = state_a.size() + state_b.size()
                            if current_total > best_total:
                                best_total = current_total
                                best_a_els = set(state_a.els)
                                best_b_els = set(state_b.els)
                            if rng.random() < 0.2:
                                pass
                            else:
                                state_a.els = old_a_els
                                state_a.sums_bits = old_a_sums_bits
                                state_a.diffs_bits = old_a_diffs_bits
                                state_b.els = old_b_els
                                state_b.sums_bits = old_b_sums_bits
                                state_b.diffs_bits = old_b_diffs_bits
                                progress = False
                        else:
                            state_a.els = old_a_els
                            state_a.sums_bits = old_a_sums_bits
                            state_a.diffs_bits = old_a_diffs_bits
                            state_b.els = old_b_els
                            state_b.sums_bits = old_b_sums_bits
                            state_b.diffs_bits = old_b_diffs_bits
                        
                        if progress:
                            break
                
                # Remove from B, add to A
                if not progress and state_b.size() > 1 and state_a.size() < MAX_SIZE:
                    b_candidates = list(state_b.els)
                    rng.shuffle(b_candidates)
                    for x_remove in b_candidates:
                        old_a_els = set(state_a.els)
                        old_a_sums_bits = state_a.sums_bits
                        old_a_diffs_bits = state_a.diffs_bits
                        old_b_els = set(state_b.els)
                        old_b_sums_bits = state_b.sums_bits
                        old_b_diffs_bits = state_b.diffs_bits
                        
                        state_b.remove(x_remove)
                        
                        added = False
                        cands_a = [x for x in all_vals if x not in state_a.els and x != x_remove]
                        rng.shuffle(cands_a)
                        for x_add in cands_a:
                            if state_a.can_add(x_add, state_b.diffs_bits):
                                state_a.add(x_add)
                                added = True
                                break
                        
                        if added:
                            progress = True
                            current_total = state_a.size() + state_b.size()
                            if current_total > best_total:
                                best_total = current_total
                                best_a_els = set(state_a.els)
                                best_b_els = set(state_b.els)
                            if rng.random() < 0.2:
                                pass
                            else:
                                state_a.els = old_a_els
                                state_a.sums_bits = old_a_sums_bits
                                state_a.diffs_bits = old_a_diffs_bits
                                state_b.els = old_b_els
                                state_b.sums_bits = old_b_sums_bits
                                state_b.diffs_bits = old_b_diffs_bits
                                progress = False
                        else:
                            state_a.els = old_a_els
                            state_a.sums_bits = old_a_sums_bits
                            state_a.diffs_bits = old_a_diffs_bits
                            state_b.els = old_b_els
                            state_b.sums_bits = old_b_sums_bits
                            state_b.diffs_bits = old_b_diffs_bits
                        
                        if progress:
                            break
        
        if not progress:
            # No moves possible. Break and let outer loop restart.
            break
            
    return best_total, best_a_els, best_b_els

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(12345)
    
    best_total = 0
    best_a = set()
    best_b = set()
    
    # Number of restarts
    num_restarts = 500  # Reduced to allow more time per restart
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
            
        state_a = SidonState()
        state_b = SidonState()
        
        # Greedy Initialization
        greedy_build(state_a, state_b, rng, deadline - 3.0)
        
        # Local Search
        ls_total, ls_a, ls_b = local_search(state_a, state_b, rng, deadline - 3.0)
        
        if ls_total > best_total:
            best_total = ls_total
            best_a = ls_a
            best_b = ls_b
            
    # Ensure nonempty
    if not best_a:
        best_a = {1}
    if not best_b:
        best_b = {2}
        
    # Convert to lists for return
    return [list(best_a), list(best_b)]

# Entry point
if __name__ == "__main__":
    # This block is for testing, the evaluator calls construct()
    pass
