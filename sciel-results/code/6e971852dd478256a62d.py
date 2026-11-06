import time
import random

N = 256
MAX_SIZE = 64
DEADLINE = 85.0  # Safety margin

class SidonPair:
    def __init__(self, A, B):
        self.A = list(A)
        self.B = list(B)
        self.sum_mask_A = 0
        self.sum_mask_B = 0
        self.diff_mask_A = 0
        self.diff_mask_B = 0
        self._rebuild()

    def _rebuild(self):
        self.sum_mask_A = 0
        self.sum_mask_B = 0
        self.diff_mask_A = 0
        self.diff_mask_B = 0
        
        for i, a in enumerate(self.A):
            for b in self.A[i:]:
                self.sum_mask_A |= (1 << (a + b))
            for b in self.A:
                d = b - a
                if d != 0:
                    self.diff_mask_A |= (1 << (d + N)) # Shift to handle negatives: 0..511

        for i, a in enumerate(self.B):
            for b in self.B[i:]:
                self.sum_mask_B |= (1 << (a + b))
            for b in self.B:
                d = b - a
                if d != 0:
                    self.diff_mask_B |= (1 << (d + N))

    def is_valid(self):
        # Check Sidon (sums unique) - handled by construction if we only add valid ones
        # Check Diff Disjointness
        if (self.diff_mask_A & self.diff_mask_B) != 0:
            return False
        # Check Size
        if len(self.A) > MAX_SIZE or len(self.B) > MAX_SIZE:
            return False
        return True

    def can_add_to_A(self, x):
        if x in self.A:
            return False
        if len(self.A) >= MAX_SIZE:
            return False
        
        # Check Sum Collisions in A
        # New sums: {a+x for a in A} U {2x}
        # These must not intersect with current sum_mask_A
        # AND the new sums must be unique among themselves?
        # If a1+x == a2+x => a1==a2, so unique if A is unique.
        # If a+x == 2x => a==x, but x not in A.
        # So we just need to check intersection with existing sum_mask_A.
        
        # Efficient check:
        # For each a in A, check if (a+x) is in sum_mask_A.
        # Also check if (2x) is in sum_mask_A.
        
        # Optimization: We can't easily do this with a single bitmask op without precomputed shifts.
        # Given small |A|, loop is fine.
        
        for a in self.A:
            if (self.sum_mask_A >> (a + x)) & 1:
                return False
        if (self.sum_mask_A >> (2 * x)) & 1:
            return False
            
        # Check Diff Disjointness with B
        # New diffs in A: {x-a for a in A} U {a-x for a in A}
        # These must not be in diff_mask_B.
        
        for a in self.A:
            d1 = x - a
            d2 = a - x
            if (self.diff_mask_B >> (d1 + N)) & 1:
                return False
            if (self.diff_mask_B >> (d2 + N)) & 1:
                return False
        return True

    def can_add_to_B(self, x):
        if x in self.B:
            return False
        if len(self.B) >= MAX_SIZE:
            return False
        
        for a in self.B:
            if (self.sum_mask_B >> (a + x)) & 1:
                return False
        if (self.sum_mask_B >> (2 * x)) & 1:
            return False
            
        for a in self.B:
            d1 = x - a
            d2 = a - x
            if (self.diff_mask_A >> (d1 + N)) & 1:
                return False
            if (self.diff_mask_A >> (d2 + N)) & 1:
                return False
        return True

    def add_to_A(self, x):
        self.A.append(x)
        for a in self.A[:-1]:
            self.sum_mask_A |= (1 << (a + x))
            d1 = x - a
            d2 = a - x
            self.diff_mask_A |= (1 << (d1 + N))
            self.diff_mask_A |= (1 << (d2 + N))
        self.sum_mask_A |= (1 << (2 * x))

    def add_to_B(self, x):
        self.B.append(x)
        for a in self.B[:-1]:
            self.sum_mask_B |= (1 << (a + x))
            d1 = x - a
            d2 = a - x
            self.diff_mask_B |= (1 << (d1 + N))
            self.diff_mask_B |= (1 << (d2 + N))
        self.sum_mask_B |= (1 << (2 * x))

    def remove_from_A(self, x):
        self.A.remove(x)
        self._rebuild() # Rebuild is safer and fast enough for |A|<20

    def remove_from_B(self, x):
        self.B.remove(x)
        self._rebuild()

    def can_swap_A(self, x_out, x_in):
        if x_out not in self.A or x_in in self.A:
            return False
        # Temporarily remove x_out
        temp_A = [a for a in self.A if a != x_out]
        # Check if x_in can be added to temp_A
        # We need to check sums and diffs against temp_A
        # And diffs against B (unchanged)
        
        # Sums in temp_A
        temp_sum_mask = 0
        for i, a in enumerate(temp_A):
            for b in temp_A[i:]:
                temp_sum_mask |= (1 << (a + b))
        
        for a in temp_A:
            if (temp_sum_mask >> (a + x_in)) & 1:
                return False
        if (temp_sum_mask >> (2 * x_in)) & 1:
            return False
            
        # Diffs in temp_A
        # We need to check if new diffs (x_in - a, a - x_in) are in B's diffs
        # Note: Diffs within A don't need to be unique, just disjoint from B.
        # So we only check against diff_mask_B.
        
        for a in temp_A:
            d1 = x_in - a
            d2 = a - x_in
            if (self.diff_mask_B >> (d1 + N)) & 1:
                return False
            if (self.diff_mask_B >> (d2 + N)) & 1:
                return False
        return True

    def do_swap_A(self, x_out, x_in):
        self.remove_from_A(x_out)
        self.add_to_A(x_in)

    def can_swap_B(self, x_out, x_in):
        if x_out not in self.B or x_in in self.B:
            return False
        temp_B = [a for a in self.B if a != x_out]
        temp_sum_mask = 0
        for i, a in enumerate(temp_B):
            for b in temp_B[i:]:
                temp_sum_mask |= (1 << (a + b))
        
        for a in temp_B:
            if (temp_sum_mask >> (a + x_in)) & 1:
                return False
        if (temp_sum_mask >> (2 * x_in)) & 1:
            return False
            
        for a in temp_B:
            d1 = x_in - a
            d2 = a - x_in
            if (self.diff_mask_A >> (d1 + N)) & 1:
                return False
            if (self.diff_mask_A >> (d2 + N)) & 1:
                return False
        return True

    def do_swap_B(self, x_out, x_in):
        self.remove_from_B(x_out)
        self.add_to_B(x_in)

    def can_move_A_to_B(self, x):
        if x not in self.A:
            return False
        # Check if x can be added to B
        if not self.can_add_to_B(x):
            return False
        # Check if removing x from A leaves A valid (Sidon)
        # Removing an element never breaks Sidon property.
        # But does it break diff disjointness?
        # Removing x from A removes some diffs from diff_mask_A.
        # This only helps disjointness (makes it easier to satisfy).
        # So if x can be added to B, the move is valid.
        return True

    def do_move_A_to_B(self, x):
        self.remove_from_A(x)
        self.add_to_B(x)

    def can_move_B_to_A(self, x):
        if x not in self.B:
            return False
        if not self.can_add_to_A(x):
            return False
        return True

    def do_move_B_to_A(self, x):
        self.remove_from_B(x)
        self.add_to_A(x)


def greedy_seed(deadline):
    """Create a good initial seed using a biased greedy approach."""
    state = SidonPair([], [])
    
    # Bias: A takes smaller numbers, B takes larger numbers
    # This helps keep differences "separated" in magnitude, though not strictly required.
    
    order_a = list(range(1, N + 1))
    order_b = list(range(N, 0, -1))
    
    random.seed(42)
    # Slight shuffle to avoid pathological cases
    random.shuffle(order_a)
    random.shuffle(order_b)
    
    # Interleaved greedy
    ia, ib = 0, 0
    while (ia < len(order_a) or ib < len(order_b)) and time.time() < deadline:
        if ia < len(order_a) and state.can_add_to_A(order_a[ia]):
            state.add_to_A(order_a[ia])
        ia += 1
        
        if ib < len(order_b) and state.can_add_to_B(order_b[ib]):
            state.add_to_B(order_b[ib])
        ib += 1
            
    return state

def local_search(state, deadline):
    """Perform local search to maximize |A| + |B|."""
    best_size = len(state.A) + len(state.B)
    best_state = state
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add elements
        if len(state.A) < MAX_SIZE:
            for x in range(1, N + 1):
                if state.can_add_to_A(x):
                    state.add_to_A(x)
                    improved = True
                    break
        if not improved and len(state.B) < MAX_SIZE:
            for x in range(1, N + 1):
                if state.can_add_to_B(x):
                    state.add_to_B(x)
                    improved = True
                    break
        
        # 2. Try to move elements
        if not improved:
            for x in list(state.A):
                if state.can_move_A_to_B(x):
                    state.do_move_A_to_B(x)
                    improved = True
                    break
        if not improved:
            for x in list(state.B):
                if state.can_move_B_to_A(x):
                    state.do_move_B_to_A(x)
                    improved = True
                    break
        
        # 3. Try swaps (remove one, add one)
        if not improved:
            # Swap in A
            if len(state.A) > 0:
                for x_out in list(state.A):
                    if time.time() > deadline:
                        break
                    for x_in in range(1, N + 1):
                        if state.can_swap_A(x_out, x_in):
                            state.do_swap_A(x_out, x_in)
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            # Swap in B
            if len(state.B) > 0:
                for x_out in list(state.B):
                    if time.time() > deadline:
                        break
                    for x_in in range(1, N + 1):
                        if state.can_swap_B(x_out, x_in):
                            state.do_swap_B(x_out, x_in)
                            improved = True
                            break
                    if improved:
                        break
        
        if improved:
            current_size = len(state.A) + len(state.B)
            if current_size > best_size:
                best_size = current_size
                best_state = state
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck for a while, try a random restart or perturbation
        if no_improve_count > max_no_improve:
            # Perturbation: Randomly remove 2 elements and let it rebuild?
            # Or just break and restart from a new seed?
            # Let's try a simple perturbation: Swap a random pair if possible, or just break.
            # Given time constraints, breaking and restarting is safer.
            break
            
    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_state = None
    best_size = 0
    
    # 1. Generate a strong seed
    seed_state = greedy_seed(deadline - 5.0)
    if seed_state:
        best_state = seed_state
        best_size = len(seed_state.A) + len(seed_state.B)
    
    # 2. Local search from seed
    if best_state and time.time() < deadline - 2.0:
        improved_state = local_search(best_state, deadline - 1.0)
        if improved_state:
            if len(improved_state.A) + len(improved_state.B) > best_size:
                best_state = improved_state
                best_size = len(improved_state.A) + len(improved_state.B)
    
    # 3. If time remains, try a few more random restarts with different biases
    if time.time() < deadline - 5.0:
        for i in range(10):
            if time.time() > deadline - 2.0:
                break
            
            # Different random seeds for greedy
            random.seed(100 + i)
            # Try different biases
            order_a = list(range(1, N + 1))
            order_b = list(range(1, N + 1))
            
            if i % 2 == 0:
                random.shuffle(order_a)
                random.shuffle(order_b)
            else:
                # Bias A to evens, B to odds?
                order_a = [x for x in order_a if x % 2 == 0] + [x for x in order_a if x % 2 == 1]
                order_b = [x for x in order_b if x % 2 == 1] + [x for x in order_b if x % 2 == 0]
                
            temp_state = SidonPair([], [])
            ia, ib = 0, 0
            while (ia < len(order_a) or ib < len(order_b)) and time.time() < deadline - 2.0:
                if ia < len(order_a) and temp_state.can_add_to_A(order_a[ia]):
                    temp_state.add_to_A(order_a[ia])
                ia += 1
                if ib < len(order_b) and temp_state.can_add_to_B(order_b[ib]):
                    temp_state.add_to_B(order_b[ib])
                ib += 1
                
            current_size = len(temp_state.A) + len(temp_state.B)
            if current_size > best_size:
                best_state = temp_state
                best_size = current_size
                
            # Quick local search on this new seed
            if time.time() < deadline - 2.0:
                improved_state = local_search(best_state, deadline - 1.0)
                if improved_state:
                    if len(improved_state.A) + len(improved_state.B) > best_size:
                        best_state = improved_state
                        best_size = len(improved_state.A) + len(improved_state.B)

    # Ensure nonempty
    if not best_state or (not best_state.A and not best_state.B):
        return [[1], [2]]
    
    return [best_state.A, best_state.B]
