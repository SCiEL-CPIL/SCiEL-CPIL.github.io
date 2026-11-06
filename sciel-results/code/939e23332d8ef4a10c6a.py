import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    N = 256

    def compute_block_sums(seq):
        """Compute all non-empty consecutive block sums as a dict {sum: (i,j)}."""
        sums = {}
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return None  # conflict
                sums[s] = (i, j)
        return sums

    def check_valid(seq):
        """Return True if all non-empty consecutive block sums are distinct."""
        return compute_block_sums(seq) is not None

    def greedy_init(order):
        """Greedy: add elements in given order if valid."""
        seq = []
        for x in order:
            new_seq = sorted(seq + [x])
            if check_valid(new_seq):
                seq = new_seq
        return seq

    def simulated_annealing(initial_seq, time_budget):
        """
        Simulated annealing on the set of selected elements.
        State: frozenset of selected values.
        Move: add one element, remove one element, or swap.
        Objective: maximize |seq| subject to distinct block sums.
        Energy = -len(seq) + penalty for conflicts (we avoid conflicts by only accepting valid states).
        Actually, we only consider valid states. Energy = -len(seq).
        Accept worse (shorter) with prob exp((len_new - len_old) / T).
        """
        if not initial_seq:
            initial_seq = greedy_init(list(range(1, N + 1)))
        
        current = set(initial_seq)
        current_seq = sorted(current)
        current_len = len(current_seq)
        
        # Verify initial is valid
        if not check_valid(current_seq):
            # Fallback to a small valid sequence
            current = set()
            for x in range(1, N + 1):
                new_seq = sorted(current + [x])
                if check_valid(new_seq):
                    current = set(new_seq)
            current_seq = sorted(current)
            current_len = len(current_seq)
        
        best_seq = current_seq[:]
        best_len = current_len
        
        T = 5.0  # initial temperature
        T_min = 0.01
        cooling = 0.9995
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > T_min:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add an element
                available = [x for x in range(1, N + 1) if x not in current]
                if not available:
                    T *= cooling
                    continue
                x = random.choice(available)
                new_current = current | {x}
                new_seq = sorted(new_current)
            elif move_type < 0.7:
                # Remove an element
                if not current:
                    T *= cooling
                    continue
                x = random.choice(list(current))
                new_current = current - {x}
                new_seq = sorted(new_current)
            else:
                # Swap
                if not current:
                    T *= cooling
                    continue
                available = [x for x in range(1, N + 1) if x not in current]
                if not available:
                    T *= cooling
                    continue
                x_out = random.choice(list(current))
                x_in = random.choice(available)
                new_current = current - {x_out} | {x_in}
                new_seq = sorted(new_current)
            
            # Check validity
            if not check_valid(new_seq):
                T *= cooling
                continue
            
            new_len = len(new_seq)
            
            # Acceptance criterion
            if new_len >= current_len:
                # Better or equal
                current = new_current
                current_seq = new_seq
                current_len = new_len
                if new_len > best_len:
                    best_len = new_len
                    best_seq = new_seq[:]
            else:
                # Worse: accept with probability
                delta = new_len - current_len  # negative
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current = new_current
                    current_seq = new_seq
                    current_len = new_len
            
            T *= cooling
        
        return best_seq

    # Phase 1: Quick greedy initializations
    best_seq = []
    best_len = 0

    # Ascending
    seq = greedy_init(list(range(1, N + 1)))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # Descending
    seq = greedy_init(list(range(N, 0, -1)))
    if len(seq) > best_len:
        best_len = len(seq)
        best_seq = seq[:]

    # A few random greedy
    for _ in range(20):
        if time.time() > deadline - 60:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        seq = greedy_init(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Phase 2: Simulated annealing from best greedy
    if time.time() < deadline - 5:
        budget = min(25, deadline - 5 - time.time())
        result = simulated_annealing(best_seq, budget)
        if len(result) > best_len:
            best_len = len(result)
            best_seq = result[:]

    # Phase 3: More annealing runs with different initializations
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        # Run multiple shorter annealing runs
        num_runs = max(1, int(remaining / 10))
        for run in range(num_runs):
            if time.time() > deadline - 5:
                break
            # Different random init
            order = list(range(1, N + 1))
            random.shuffle(order)
            init_seq = greedy_init(order)
            # Mix with current best
            if random.random() < 0.5 and best_seq:
                # Start from best, perturb
                perturb = list(best_seq)
                for _ in range(random.randint(1, 5)):
                    if perturb:
                        idx = random.randint(0, len(perturb) - 1)
                        removed = perturb.pop(idx)
                        available = [x for x in range(1, N + 1) if x not in set(perturb)]
                        if available:
                            perturb.append(random.choice(available))
                init_seq = sorted(set(perturb))
            
            run_budget = min(8, deadline - 5 - time.time())
            if run_budget < 1:
                break
            result = simulated_annealing(init_seq, run_budget)
            if len(result) > best_len:
                best_len = len(result)
                best_seq = result[:]

    return best_seq

