import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    class State:
        def __init__(self, A, B):
            self.A = A
            self.B = B
            self.sums_A = set()
            self.sums_B = set()
            self.diffs_A = set()
            self.diffs_B = set()
            self._rebuild()
            
        def _rebuild(self):
            self.sums_A = set()
            self.sums_B = set()
            self.diffs_A = set()
            self.diffs_B = set()
            
            for i in range(len(self.A)):
                for j in range(i, len(self.A)):
                    self.sums_A.add(self.A[i] + self.A[j])
                    if i != j:
                        self.diffs_A.add(abs(self.A[i] - self.A[j]))
            
            for i in range(len(self.B)):
                for j in range(i, len(self.B)):
                    self.sums_B.add(self.B[i] + self.B[j])
                    if i != j:
                        self.diffs_B.add(abs(self.B[i] - self.B[j]))
                        
        def can_add(self, x, target, other_diffs):
            """Check if x can be added to target set."""
            if target is self.A:
                sums = self.sums_A
            else:
                sums = self.sums_B
                
            if 2 * x in sums:
                return False
            for a in target:
                if x + a in sums:
                    return False
                d = abs(x - a)
                if d in other_diffs:
                    return False
            return True

        def add(self, x, target):
            """Add x to target set and update state."""
            if target is self.A:
                sums = self.sums_A
                diffs = self.diffs_A
                other_diffs = self.diffs_B
            else:
                sums = self.sums_B
                diffs = self.diffs_B
                other_diffs = self.diffs_A
                
            sums.add(2 * x)
            for a in target:
                sums.add(x + a)
                diffs.add(abs(x - a))
            target.append(x)
            
        def remove(self, x, target):
            """Remove x from target set and update state."""
            if target is self.A:
                sums = self.sums_A
                diffs = self.diffs_A
            else:
                sums = self.sums_B
                diffs = self.diffs_B
                
            target.remove(x)
            # Rebuilding is safer and faster for small sets than incremental removal
            # But for local search, we might want incremental. 
            # Given n=256 and size ~16, O(k^2) rebuild is ~256 ops. 
            # Incremental removal requires removing all pairs involving x.
            # Let's stick to rebuild for simplicity and correctness in this edit, 
            # but optimize by only rebuilding the affected set if possible? 
            # No, diffs are coupled. Full rebuild is safest.
            self._rebuild()

    def greedy_construct(order, prefer_A_first=True):
        """Greedy construction with given order."""
        A = []
        B = []
        state = State(A, B)
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            if prefer_A_first:
                if state.can_add(x, A, state.diffs_B):
                    state.add(x, A)
                elif state.can_add(x, B, state.diffs_A):
                    state.add(x, B)
            else:
                if state.can_add(x, B, state.diffs_A):
                    state.add(x, B)
                elif state.can_add(x, A, state.diffs_B):
                    state.add(x, A)
                    
        return A, B

    def local_search(A, B):
        """
        Local search using incremental state updates where possible.
        Tries 1-add, 1-move, and 2-swap moves.
        """
        A = A[:]
        B = B[:]
        state = State(A, B)
        
        max_iterations = 500
        no_improve_count = 0
        
        for _ in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            best_move = None
            best_gain = 0
            
            # 1. Try to add any element
            # We check all 1..n
            # Optimization: Only check elements not in A or B
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                if state.can_add(x, A, state.diffs_B):
                    # Gain 1
                    if best_gain < 1:
                        best_gain = 1
                        best_move = ('add', x, A)
                elif state.can_add(x, B, state.diffs_A):
                    if best_gain < 1:
                        best_gain = 1
                        best_move = ('add', x, B)
                        
            # 2. Try to move an element from A to B or B to A
            # This might allow other elements to be added later, but immediately size is same.
            # However, it might unblock future adds. 
            # We don't have a lookahead, so we only accept moves that allow an immediate add?
            # Or we can accept moves that reduce "conflict score"?
            # Let's stick to moves that result in a valid state and potentially allow an add.
            # Actually, a move that doesn't change size is only useful if it enables a subsequent add.
            # We can simulate: Move x from A to B. Then try to add y.
            # This is expensive. 
            # Alternative: Just try 2-swaps which change composition.
            
            # 3. Try 2-Swaps: Remove a from A, remove b from B, add c to A, add d to B?
            # Too complex for now. Let's try "Remove a from A, Add b to A" (1-swap within set)
            # or "Remove a from A, Add b to B" (Move).
            
            # Let's try a simpler heuristic: 
            # If no add is possible, try a "swap" that might open up space.
            # We measure "potential" by how many new elements can be added after the swap.
            # This is O(N^2) per swap, which is too slow.
            
            # Instead, let's do a "randomized hill climbing" on the set of elements.
            # Pick a random element in A, remove it. Pick a random element not in A/B, add to A.
            # If valid, keep. If not, revert.
            
            if best_move:
                op, x, target = best_move
                state.add(x, target)
                improved = True
                no_improve_count = 0
            else:
                # No direct add. Try a random swap to escape local optimum.
                # Remove a random element from A or B, and try to add a random new element to the same set.
                # Or move an element between sets.
                
                # Try moving an element from A to B
                if A and random.random() < 0.5:
                    a_idx = random.randint(0, len(A) - 1)
                    a = A[a_idx]
                    # Remove a from A
                    A.remove(a)
                    state._rebuild()
                    
                    # Try to add a to B
                    if state.can_add(a, B, state.diffs_A):
                        state.add(a, B)
                        improved = True
                        no_improve_count = 0
                    else:
                        # Revert: put a back in A
                        A.append(a)
                        state._rebuild()
                        no_improve_count += 1
                else:
                    # Try moving from B to A
                    if B and random.random() < 0.5:
                        b_idx = random.randint(0, len(B) - 1)
                        b = B[b_idx]
                        B.remove(b)
                        state._rebuild()
                        
                        if state.can_add(b, A, state.diffs_B):
                            state.add(b, A)
                            improved = True
                            no_improve_count = 0
                        else:
                            B.append(b)
                            state._rebuild()
                            no_improve_count += 1
                    else:
                        # Try replacing an element in A with a new one
                        if A:
                            a_idx = random.randint(0, len(A) - 1)
                            a = A[a_idx]
                            A.remove(a)
                            state._rebuild()
                            
                            # Try to add a random new element to A
                            new_x = random.randint(1, n)
                            if new_x not in A and new_x not in B:
                                if state.can_add(new_x, A, state.diffs_B):
                                    state.add(new_x, A)
                                    improved = True
                                    no_improve_count = 0
                                else:
                                    # Revert
                                    A.append(a)
                                    state._rebuild()
                                    no_improve_count += 1
                            else:
                                A.append(a)
                                state._rebuild()
                                no_improve_count += 1
                        else:
                            no_improve_count += 1

            if no_improve_count > 50:
                break
                
        return A, B

    def create_biased_order(bias_strength=0.3):
        """Create a biased ordering with smaller numbers preferred."""
        order = list(range(1, n + 1))
        for i in range(len(order) - 1):
            if random.random() < bias_strength * (1 - i / len(order)):
                j = random.randint(i, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order

    # Main search loop
    restarts = 0
    while time.time() - start_time < time_limit:
        # Vary bias strength for diversity
        bias = 0.2 + random.random() * 0.3
        order = create_biased_order(bias)
        
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(order, prefer_A)
        
        # Local improvement pass
        A, B = local_search(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
        
        restarts += 1
        
        # If we've found a good solution, do more careful refinement
        if best_size >= 30 and restarts % 5 == 0:
            # Try to improve the best solution found so far
            A, B = local_search(best_A, best_B)
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A = A[:]
                best_B = B[:]

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
