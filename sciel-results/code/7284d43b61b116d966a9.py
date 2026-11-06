import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit

    n = 16
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)  # 560
    triple_to_id = {t: i for i, t in enumerate(triples)}

    # Precompute K(2,2,2) configurations
    k222_triples = []
    triple_to_k222s = [[] for _ in range(num_triples)]

    def get_pair_partitions(six):
        """Generate all 15 ways to partition 6 vertices into 3 pairs."""
        v = list(six)
        parts = []
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            for j in range(1, 4):
                pair2 = (rem[0], rem[j])
                if j == 1:
                    pair3 = (rem[2], rem[3])
                elif j == 2:
                    pair3 = (rem[1], rem[3])
                else:
                    pair3 = (rem[1], rem[2])
                parts.append([pair1, pair2, pair3])
        return parts

    for six_subset in combinations(range(n), 6):
        for partition in get_pair_partitions(six_subset):
            t_ids = []
            for x in partition[0]:
                for y in partition[1]:
                    for z in partition[2]:
                        t = tuple(sorted((x, y, z)))
                        t_id = triple_to_id[t]
                        t_ids.append(t_id)
            k_idx = len(k222_triples)
            k222_triples.append(t_ids)
            for t_id in t_ids:
                triple_to_k222s[t_id].append(k_idx)

    num_k222s = len(k222_triples)
    triple_k222_count = [len(triple_to_k222s[t]) for t in range(num_triples)]

    class State:
        def __init__(self):
            self.present = [False] * num_triples
            self.k_counts = [0] * num_k222s
            self.size = 0

        def can_add(self, t_id):
            if self.present[t_id]:
                return True
            for k_idx in triple_to_k222s[t_id]:
                if self.k_counts[k_idx] == 7:
                    return False
            return True

        def add(self, t_id):
            if not self.can_add(t_id):
                return False
            self.present[t_id] = True
            self.size += 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] += 1
            return True

        def remove(self, t_id):
            if not self.present[t_id]:
                return
            self.present[t_id] = False
            self.size -= 1
            for k_idx in triple_to_k222s[t_id]:
                self.k_counts[k_idx] -= 1

        def present_list(self):
            return [i for i in range(num_triples) if self.present[i]]

        def copy_state(self):
            s = State()
            s.present = self.present[:]
            s.k_counts = self.k_counts[:]
            s.size = self.size
            return s

    def polish(state, max_iters=100000, tight=False):
        """Try to add any missing triple that fits. Tight mode does multiple passes."""
        improved = True
        iter_count = 0
        max_passes = 5 if tight else 2
        pass_num = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            pass_num += 1
            if pass_num > max_passes:
                break
            # Get missing triples sorted by constraint count (ascending)
            missing = [(triple_k222_count[t], t) for t in range(num_triples) if not state.present[t]]
            if tight and pass_num > 1:
                # In tight mode, shuffle the order for subsequent passes
                random.shuffle(missing)
            else:
                missing.sort()
            for _, t_id in missing:
                if state.add(t_id):
                    improved = True
            # Random additions
            n_random = 500 if tight else 200
            for _ in range(n_random):
                if time.time() > DEADLINE:
                    break
                t_id = random.randint(0, num_triples - 1)
                if not state.present[t_id] and state.add(t_id):
                    improved = True

    def kick_local_search(state, max_kicks=500):
        """Kick-based local search: remove random present triples to unblock additions."""
        best_size = state.size
        best_present = state.present[:]
        best_k_counts = state.k_counts[:]
        
        for kick_iter in range(max_kicks):
            if time.time() > DEADLINE:
                break
            
            # Try to add some missing triples first
            polish(state, max_iters=10000)
            
            if state.size > best_size:
                best_size = state.size
                best_present = state.present[:]
                best_k_counts = state.k_counts[:]
            
            # Find a blocked triple to kick
            present = state.present_list()
            if not present:
                break
            
            # Pick a random present triple to remove (kick)
            t_out = random.choice(present)
            state.remove(t_out)
            
            # After removing, try to add multiple missing triples
            improved = True
            add_count = 0
            while improved and add_count < 10:
                improved = False
                add_count += 1
                # Try adding random missing triples
                for _ in range(50):
                    if time.time() > DEADLINE:
                        break
                    t_in = random.randint(0, num_triples - 1)
                    if not state.present[t_in] and state.add(t_in):
                        improved = True
                        break
                # Also try systematic addition
                for t_in in range(num_triples):
                    if not state.present[t_in] and state.add(t_in):
                        improved = True
                        break
            
            # Polish after kick
            polish(state, max_iters=50000)
            
            if state.size > best_size:
                best_size = state.size
                best_present = state.present[:]
                best_k_counts = state.k_counts[:]
            
            # If we didn't improve, consider reverting to best
            if state.size < best_size - 2 and random.random() < 0.5:
                # Revert to best state
                for i in range(num_triples):
                    if state.present[i]:
                        state.remove(i)
                for i in range(num_triples):
                    if best_present[i]:
                        state.add(i)
        
        # Restore best state
        if best_size > state.size:
            for i in range(num_triples):
                if state.present[i]:
                    state.remove(i)
            for i in range(num_triples):
                if best_present[i]:
                    state.add(i)
        
        return best_size, best_present

    def simulated_annealing(state, initial_temp=15.0, cooling=0.9995, max_iters=50000):
        """Simulated annealing with kicking."""
        temp = initial_temp
        current_size = state.size
        best_size = state.size
        best_present = state.present[:]
        
        for it in range(max_iters):
            if time.time() > DEADLINE:
                break
            if temp < 0.05:
                break
            
            # Choose move type: swap (80%), kick (15%), double-swap (5%)
            move_type = random.random()
            
            if move_type < 0.80:
                # Standard swap: remove one, add one
                present = state.present_list()
                if not present:
                    break
                t_out = random.choice(present)
                t_in = random.randint(0, num_triples - 1)
                
                if t_in == t_out or state.present[t_in]:
                    continue
                
                state.remove(t_out)
                
                if state.add(t_in):
                    new_size = state.size
                    delta = new_size - current_size
                    if delta >= 0 or random.random() < np.exp(delta / temp):
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_present = state.present[:]
                    else:
                        state.remove(t_in)
                        state.add(t_out)
                else:
                    # Try a few alternatives
                    found = False
                    for _ in range(3):
                        t_in2 = random.randint(0, num_triples - 1)
                        if t_in2 != t_out and not state.present[t_in2]:
                            if state.add(t_in2):
                                new_size = state.size
                                delta = new_size - current_size
                                if delta >= 0 or random.random() < np.exp(delta / temp):
                                    current_size = new_size
                                    if current_size > best_size:
                                        best_size = current_size
                                        best_present = state.present[:]
                                else:
                                    state.remove(t_in2)
                                    state.add(t_out)
                                found = True
                                break
                    if not found:
                        state.add(t_out)
            
            elif move_type < 0.95:
                # Kick: remove one, try to add multiple
                present = state.present_list()
                if not present:
                    break
                t_out = random.choice(present)
                state.remove(t_out)
                
                # Try to add several missing triples
                added_any = False
                for _ in range(20):
                    t_in = random.randint(0, num_triples - 1)
                    if not state.present[t_in] and state.add(t_in):
                        added_any = True
                        break
                
                new_size = state.size
                delta = new_size - current_size
                if delta >= 0 or random.random() < np.exp(delta / temp):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_present = state.present[:]
                else:
                    # Revert: remove what we added, put back t_out
                    for t_in in range(num_triples):
                        if state.present[t_in] and not best_present[t_in]:
                            state.remove(t_in)
                    state.add(t_out)
                    current_size = state.size
            
            else:
                # Double swap: remove two, add two
                present = state.present_list()
                if len(present) < 2:
                    break
                t_out1, t_out2 = random.sample(present, 2)
                state.remove(t_out1)
                state.remove(t_out2)
                
                added = 0
                for _ in range(30):
                    t_in = random.randint(0, num_triples - 1)
                    if not state.present[t_in] and state.add(t_in):
                        added += 1
                        if added >= 2:
                            break
                
                new_size = state.size
                delta = new_size - current_size
                if delta >= 0 or random.random() < np.exp(delta / temp):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_present = state.present[:]
                else:
                    # Revert
                    for t_in in range(num_triples):
                        if state.present[t_in] and not best_present[t_in]:
                            state.remove(t_in)
                    state.add(t_out1)
                    state.add(t_out2)
                    current_size = state.size
            
            temp *= cooling
        
        # Restore best
        if best_size > state.size:
            for i in range(num_triples):
                if state.present[i]:
                    state.remove(i)
            for i in range(num_triples):
                if best_present[i]:
                    state.add(i)
        
        return best_size, best_present

    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        # Sort by constraint count with some randomness
        base_order = sorted(range(num_triples), key=lambda t: (triple_k222_count[t], local_rng.random()))
        for t_id in base_order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state

    best_size = 0
    best_present = None
    rng = random.Random(42)

    # Phase 1: Multiple greedy initializations
    for attempt in range(5):
        if time.time() > DEADLINE - 30:
            break
        state = greedy_init(42 + attempt * 7919)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]

    # Phase 2: Main search loop with SA and kick-based local search
    restart_count = 0
    while time.time() < DEADLINE - 10:
        restart_count += 1
        seed = 42 + restart_count * 7919

        if restart_count <= 2:
            # Start from best and perturb
            state = State()
            for i in range(num_triples):
                if best_present and best_present[i]:
                    state.add(i)

            # Perturb: remove some random present triples
            n_perturb = min(20, state.size // 8 + 5)
            present = state.present_list()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_list()

            polish(state)
        else:
            # Fresh greedy with different seed
            state = greedy_init(seed)

        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]

        # Simulated annealing
        sa_size, sa_present = simulated_annealing(state)
        if sa_size > best_size:
            best_size = sa_size
            best_present = sa_present[:]

        # Kick-based local search
        state2 = State()
        for i in range(num_triples):
            if best_present[i]:
                state2.add(i)
        kick_size, kick_present = kick_local_search(state2, max_kicks=200)
        if kick_size > best_size:
            best_size = kick_size
            best_present = kick_present[:]

        # Polish
        polish(state2)
        if state2.size > best_size:
            best_size = state2.size
            best_present = state2.present[:]

    # Phase 3: Final tight polish on the best solution
    if best_present is not None and time.time() < DEADLINE - 2:
        final_state = State()
        for i in range(num_triples):
            if best_present[i]:
                final_state.add(i)
        polish(final_state, max_iters=200000, tight=True)
        if final_state.size > best_size:
            best_size = final_state.size
            best_present = final_state.present[:]

    return [list(triples[t_id]) for t_id in range(num_triples) if best_present and best_present[t_id]]
