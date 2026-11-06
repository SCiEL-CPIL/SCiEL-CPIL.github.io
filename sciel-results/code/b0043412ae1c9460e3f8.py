import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0

class SidonState:
    def __init__(self):
        self.A = []
        self.B = []
        self.sums_A = set()
        self.sums_B = set()
        self.diff_counts_A = {}
        self.diff_counts_B = {}
        self.diffs_A = set()
        self.diffs_B = set()
        
    def can_add_to_A(self, x):
        if 2*x in self.sums_A:
            return False
        for y in self.A:
            if x + y in self.sums_A:
                return False
            d = abs(x - y)
            if d in self.diffs_A or d in self.diffs_B:
                return False
        return True

    def can_add_to_B(self, x):
        if 2*x in self.sums_B:
            return False
        for y in self.B:
            if x + y in self.sums_B:
                return False
            d = abs(x - y)
            if d in self.diffs_A or d in self.diffs_B:
                return False
        return True

    def add_to_A(self, x):
        self.A.append(x)
        self.sums_A.add(2*x)
        for y in self.A[:-1]:
            s = x + y
            self.sums_A.add(s)
            d = abs(x - y)
            if d not in self.diff_counts_A:
                self.diff_counts_A[d] = 0
                self.diffs_A.add(d)
            self.diff_counts_A[d] += 1

    def add_to_B(self, x):
        self.B.append(x)
        self.sums_B.add(2*x)
        for y in self.B[:-1]:
            s = x + y
            self.sums_B.add(s)
            d = abs(x - y)
            if d not in self.diff_counts_B:
                self.diff_counts_B[d] = 0
                self.diffs_B.add(d)
            self.diff_counts_B[d] += 1

    def remove_from_A(self, x):
        # Find index
        idx = self.A.index(x)
        self.A.pop(idx)
        self.sums_A.remove(2*x)
        for y in self.A:
            s = x + y
            self.sums_A.remove(s)
            d = abs(x - y)
            self.diff_counts_A[d] -= 1
            if self.diff_counts_A[d] == 0:
                self.diffs_A.remove(d)
                del self.diff_counts_A[d]

    def remove_from_B(self, x):
        idx = self.B.index(x)
        self.B.pop(idx)
        self.sums_B.remove(2*x)
        for y in self.B:
            s = x + y
            self.sums_B.remove(s)
            d = abs(x - y)
            self.diff_counts_B[d] -= 1
            if self.diff_counts_B[d] == 0:
                self.diffs_B.remove(d)
                del self.diff_counts_B[d]

    def size(self):
        return len(self.A) + len(self.B)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_size = 0
    best_A = []
    best_B = []
    
    rng = random.Random(42)

    def greedy_build():
        state = SidonState()
        # Randomized greedy
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            # Try to add to smaller set first, or random
            if len(state.A) < len(state.B):
                target = 0
            elif len(state.B) < len(state.A):
                target = 1
            else:
                target = rng.choice([0, 1])
                
            if target == 0:
                if len(state.A) < MAX_EACH and state.can_add_to_A(x):
                    state.add_to_A(x)
                elif len(state.B) < MAX_EACH and state.can_add_to_B(x):
                    state.add_to_B(x)
            else:
                if len(state.B) < MAX_EACH and state.can_add_to_B(x):
                    state.add_to_B(x)
                elif len(state.A) < MAX_EACH and state.can_add_to_A(x):
                    state.add_to_A(x)
                    
        return state

    def local_search(state, max_time):
        current_state = state
        current_size = current_state.size()
        best_local_size = current_size
        best_local_A = list(current_state.A)
        best_local_B = list(current_state.B)
        
        T = 1.0
        cooling = 0.9995
        steps = 0
        no_improve = 0
        
        while time.time() < max_time:
            if T < 0.01:
                break
                
            move_type = rng.random()
            
            if move_type < 0.4:
                # ADD
                # Pick a random candidate
                all_elements = set(range(1, N + 1))
                used = set(current_state.A) | set(current_state.B)
                candidates = list(all_elements - used)
                if candidates:
                    x = rng.choice(candidates)
                    target = rng.choice([0, 1])
                    
                    if target == 0 and len(current_state.A) < MAX_EACH:
                        if current_state.can_add_to_A(x):
                            current_state.add_to_A(x)
                            new_size = current_state.size()
                            delta = new_size - current_size
                            if delta > 0 or rng.random() < pow(2.71828, delta / T):
                                current_size = new_size
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_A = list(current_state.A)
                                    best_local_B = list(current_state.B)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                steps += 1
                                T *= cooling
                                continue
                    elif target == 1 and len(current_state.B) < MAX_EACH:
                        if current_state.can_add_to_B(x):
                            current_state.add_to_B(x)
                            new_size = current_state.size()
                            delta = new_size - current_size
                            if delta > 0 or rng.random() < pow(2.71828, delta / T):
                                current_size = new_size
                                if current_size > best_local_size:
                                    best_local_size = current_size
                                    best_local_A = list(current_state.A)
                                    best_local_B = list(current_state.B)
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                steps += 1
                                T *= cooling
                                continue

            elif move_type < 0.6:
                # REMOVE
                if current_state.A:
                    x = rng.choice(current_state.A)
                    current_state.remove_from_A(x)
                    new_size = current_state.size()
                    delta = new_size - current_size
                    if rng.random() < pow(2.71828, delta / T):
                        current_size = new_size
                        no_improve += 1
                        steps += 1
                        T *= cooling
                        continue
                    else:
                        current_state.add_to_A(x)
                elif current_state.B:
                    x = rng.choice(current_state.B)
                    current_state.remove_from_B(x)
                    new_size = current_state.size()
                    delta = new_size - current_size
                    if rng.random() < pow(2.71828, delta / T):
                        current_size = new_size
                        no_improve += 1
                        steps += 1
                        T *= cooling
                        continue
                    else:
                        current_state.add_to_B(x)

            else:
                # EXCHANGE (within set)
                # Remove x from A, add y to A
                if current_state.A:
                    x = rng.choice(current_state.A)
                    # Temporarily remove
                    current_state.remove_from_A(x)
                    
                    # Find candidate y
                    all_elements = set(range(1, N + 1))
                    used = set(current_state.A) | set(current_state.B)
                    candidates = list(all_elements - used)
                    found = False
                    if candidates:
                        # Try a few random ones
                        for _ in range(5):
                            if not candidates: break
                            y = rng.choice(candidates)
                            if current_state.can_add_to_A(y):
                                current_state.add_to_A(y)
                                found = True
                                break
                            
                    if not found:
                        # Revert
                        current_state.add_to_A(x)
                    else:
                        # Size is same, so delta = 0. Always accept? 
                        # Or accept with prob 1 to explore.
                        current_size = current_state.size()
                        no_improve += 1
                        steps += 1
                        T *= cooling
                        continue

            # If no move was made or accepted, cool down slightly
            T *= 0.99
            
            if no_improve > 1000:
                break

        return best_local_A, best_local_B, best_local_size

    # Main loop
    while time.time() < deadline - 2.0:
        state = greedy_build()
        
        # Limit time for local search per restart
        ls_deadline = time.time() + 3.0
        if ls_deadline > deadline - 2.0:
            ls_deadline = deadline - 2.0
            
        A, B, size = local_search(state, ls_deadline)
        
        if size > best_size:
            best_size = size
            best_A = list(A)
            best_B = list(B)
            
        if best_size >= 2 * MAX_EACH:
            break

    return [best_A, best_B]

if __name__ == "__main__":
    print(construct())
