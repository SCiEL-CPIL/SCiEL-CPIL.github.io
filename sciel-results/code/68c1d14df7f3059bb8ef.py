import time
import random
import sys

N = 256
MAX_SIZE = 64
DEADLINE = 50.0  # Safety margin for 60s limit

class SidonState:
    """
    Tracks elements, pair sums, and nonzero differences for a Sidon set.
    """
    def __init__(self):
        self.els = set()
        self.sums = set()
        self.diffs = set()
        
    def can_add(self, x, other_diffs=None):
        if x in self.els:
            return False
        # Check sums
        if x + x in self.sums:
            return False
        for a in self.els:
            if a + x in self.sums:
                return False
        # Check diffs
        for a in self.els:
            d = x - a
            if d in self.diffs:
                return False
            if other_diffs is not None and d in other_diffs:
                return False
        return True

    def add(self, x):
        self.els.add(x)
        self.sums.add(x + x)
        for a in list(self.els):
            if a != x:
                self.sums.add(a + x)
                self.diffs.add(x - a)
                self.diffs.add(a - x)

    def remove(self, x):
        if x not in self.els:
            return
        self.els.discard(x)
        self.sums.discard(x + x)
        for a in list(self.els):
            self.sums.discard(a + x)
            self.diffs.discard(x - a)
            self.diffs.discard(a - x)

    def size(self):
        return len(self.els)

    def copy(self):
        new_state = SidonState()
        new_state.els = set(self.els)
        new_state.sums = set(self.sums)
        new_state.diffs = set(self.diffs)
        return new_state

def greedy_build(order, other_state=None, max_size=MAX_SIZE):
    state = SidonState()
    for x in order:
        if state.size() >= max_size:
            break
        if state.can_add(x, other_state.diffs if other_state else None):
            state.add(x)
    return state

def local_search(A_state, B_state, deadline, rng):
    """
    Local search with Add, Swap, and Move operations.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_total = A_state.size() + B_state.size()
        
        # 1. Try Pure Additions
        # Add to A
        if A_state.size() < MAX_SIZE:
            candidates = [x for x in range(1, N + 1) if x not in A_state.els]
            rng.shuffle(candidates)
            for x in candidates:
                if A_state.can_add(x, B_state.diffs):
                    A_state.add(x)
                    improved = True
                    break
        
        # Add to B
        if B_state.size() < MAX_SIZE:
            candidates = [x for x in range(1, N + 1) if x not in B_state.els]
            rng.shuffle(candidates)
            for x in candidates:
                if B_state.can_add(x, A_state.diffs):
                    B_state.add(x)
                    improved = True
                    break

        # 2. Try Swaps (Remove one, Add one) if stuck or to optimize
        if not improved:
            # Swap in A
            if A_state.size() > 0 and A_state.size() < MAX_SIZE:
                a_list = list(A_state.els)
                rng.shuffle(a_list)
                for a_out in a_list:
                    A_state.remove(a_out)
                    # Try to add something new to A
                    candidates = [x for x in range(1, N + 1) if x not in A_state.els and x != a_out]
                    rng.shuffle(candidates)
                    added = False
                    for x_in in candidates:
                        if A_state.can_add(x_in, B_state.diffs):
                            A_state.add(x_in)
                            added = True
                            improved = True
                            break
                    if not added:
                        # Revert
                        A_state.add(a_out)
                    else:
                        break # Restart loop with new state
            
            # Swap in B
            if not improved and B_state.size() > 0 and B_state.size() < MAX_SIZE:
                b_list = list(B_state.els)
                rng.shuffle(b_list)
                for b_out in b_list:
                    B_state.remove(b_out)
                    candidates = [x for x in range(1, N + 1) if x not in B_state.els and x != b_out]
                    rng.shuffle(candidates)
                    added = False
                    for x_in in candidates:
                        if B_state.can_add(x_in, A_state.diffs):
                            B_state.add(x_in)
                            added = True
                            improved = True
                            break
                    if not added:
                        B_state.add(b_out)
                    else:
                        break

        # 3. Try Moves (Move element from A to B or B to A)
        if not improved:
            # Move from A to B
            if A_state.size() > 0 and B_state.size() < MAX_SIZE:
                a_list = list(A_state.els)
                rng.shuffle(a_list)
                for a in a_list:
                    A_state.remove(a)
                    if B_state.can_add(a, A_state.diffs):
                        B_state.add(a)
                        improved = True
                        break
                    else:
                        A_state.add(a) # Revert

            # Move from B to A
            if not improved and B_state.size() > 0 and A_state.size() < MAX_SIZE:
                b_list = list(B_state.els)
                rng.shuffle(b_list)
                for b in b_list:
                    B_state.remove(b)
                    if A_state.can_add(b, B_state.diffs):
                        A_state.add(b)
                        improved = True
                        break
                    else:
                        B_state.add(b) # Revert

    return A_state, B_state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_A = set()
    best_B = set()
    best_total = 0
    
    rng = random.Random(42)
    
    # Multiple restarts
    max_restarts = 100
    for trial in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random orders
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        rng.shuffle(order_a)
        rng.shuffle(order_b)
        
        # Alternate building order to break symmetry
        if trial % 2 == 0:
            A_state = greedy_build(order_a, other_state=None)
            B_state = greedy_build(order_b, other_state=A_state)
        else:
            B_state = greedy_build(order_b, other_state=None)
            A_state = greedy_build(order_a, other_state=B_state)
        
        # Local search
        A_state, B_state = local_search(A_state, B_state, deadline - 2.0, rng)
        
        total = A_state.size() + B_state.size()
        if total > best_total:
            best_total = total
            best_A = set(A_state.els)
            best_B = set(B_state.els)
            
            # Early stop if we hit theoretical bound? 
            # Theoretical max is around 32. If we hit 30+, we are good.
            if best_total >= 30:
                # We can try a few more to see if we can push to 32, but 30 is already great.
                # Let's run a few more restarts to be sure.
                pass

    # Fallback if nothing found (should not happen)
    if not best_A:
        best_A = {1}
    if not best_B:
        best_B = {2}
        
    # Ensure caps
    if len(best_A) > MAX_SIZE:
        best_A = set(list(best_A)[:MAX_SIZE])
    if len(best_B) > MAX_SIZE:
        best_B = set(list(best_B)[:MAX_SIZE])

    return [sorted(best_A), sorted(best_B)]

if __name__ == "__main__":
    A, B = construct()
    print(f"Sizes: {len(A)}, {len(B)}, Total: {len(A)+len(B)}")
