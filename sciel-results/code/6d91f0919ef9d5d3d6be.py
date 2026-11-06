import time
import random
from itertools import combinations

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
        # Fix v[0] paired with v[1..5]
        for i in range(1, 6):
            pair1 = (v[0], v[i])
            rem = [v[j] for j in range(1, 6) if j != i]
            # Pair rem[0] with rem[1], rem[2], or rem[3]
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
    
    num_k222s = len(k222_triples)  # 120120
    
    # Count how many K(2,2,2)s each triple is in
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
        
        def missing_list(self):
            return [i for i in range(num_triples) if not self.present[i]]
    
    def polish(state, max_iters=50000):
        """Try to add any missing triple that fits."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < DEADLINE:
            improved = False
            iter_count += 1
            # Try to add random missing triples
            for _ in range(300):
                if time.time() > DEADLINE:
                    break
                t_id = random.randint(0, num_triples - 1)
                if not state.present[t_id] and state.add(t_id):
                    improved = True
    
    def polish_full(state):
        """Systematically try to add all possible missing triples."""
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for t_id in range(num_triples):
                if not state.present[t_id]:
                    if state.add(t_id):
                        improved = True
    
    def random_1swap(state):
        """Remove one present, add one missing."""
        present = state.present_list()
        if not present:
            return False
        t_out = random.choice(present)
        # Find a missing triple
        for _ in range(20):
            t_in = random.randint(0, num_triples - 1)
            if not state.present[t_in]:
                break
        else:
            return False
        
        state.remove(t_out)
        if state.add(t_in):
            return True
        else:
            state.add(t_out)
            return False
    
    def random_2swap(state):
        """Remove two present, add two missing."""
        present = state.present_list()
        if len(present) < 2:
            return False
        t_out1, t_out2 = random.sample(present, 2)
        
        missing = []
        for _ in range(40):
            t = random.randint(0, num_triples - 1)
            if not state.present[t]:
                missing.append(t)
            if len(missing) >= 2:
                break
        if len(missing) < 2:
            return False
        t_in1, t_in2 = missing
        
        state.remove(t_out1)
        state.remove(t_out2)
        added1 = state.add(t_in1)
        added2 = state.add(t_in2)
        if added1 and added2:
            return True
        else:
            # Revert
            if added1:
                state.remove(t_in1)
            if added2:
                state.remove(t_in2)
            state.add(t_out2)
            state.add(t_out1)
            return False
    
    def random_3swap(state):
        """Remove three present, add three missing."""
        present = state.present_list()
        if len(present) < 3:
            return False
        t_outs = random.sample(present, 3)
        
        missing = []
        for _ in range(60):
            t = random.randint(0, num_triples - 1)
            if not state.present[t]:
                missing.append(t)
            if len(missing) >= 3:
                break
        if len(missing) < 3:
            return False
        t_ins = missing[:3]
        
        for t in t_outs:
            state.remove(t)
        added = []
        for t in t_ins:
            if state.add(t):
                added.append(t)
            else:
                break
        if len(added) == 3:
            return True
        else:
            # Revert
            for t in added:
                state.remove(t)
            for t in t_outs:
                state.add(t)
            return False
    
    def sat_based_local_search(state, max_iters=5000):
        """
        Max-SAT style local search: identify triples that are 'blocking'
        (in many nearly-saturated K(2,2,2)s) and try to swap them out
        for triples that are in fewer constraints.
        """
        if time.time() > DEADLINE:
            return False
        
        improved = False
        present = state.present_list()
        if not present:
            return False
        
        # Compute for each present triple: how many K(2,2,2)s it's in with count 7 (blocking)
        # vs count < 7 (non-blocking)
        blocking_score = {}
        for t_id in present:
            blocking = 0
            for k_idx in triple_to_k222s[t_id]:
                if state.k_counts[k_idx] >= 7:
                    blocking += 1
            blocking_score[t_id] = blocking
        
        # For each missing triple: how many K(2,2,2)s it could join (count < 7)
        # We want to find swaps that reduce total blocking
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            
            # Pick a present triple with high blocking score
            sorted_present = sorted(present, key=lambda t: blocking_score.get(t, 0), reverse=True)
            if not sorted_present:
                break
            
            # Try a few of the most blocking triples
            candidates_out = sorted_present[:min(5, len(sorted_present))]
            
            swap_found = False
            for t_out in candidates_out:
                if time.time() > DEADLINE:
                    break
                
                # Find a missing triple that would be less blocking
                best_t_in = None
                best_gain = 0
                
                # Sample some missing triples
                for _ in range(30):
                    t_in = random.randint(0, num_triples - 1)
                    if state.present[t_in]:
                        continue
                    
                    # Estimate gain: removing t_out frees up its K(2,2,2)s,
                    # adding t_in uses up some K(2,2,2)s
                    # Net benefit if t_in is in fewer tight K(2,2,2)s than t_out
                    t_out_tight = sum(1 for k in triple_to_k222s[t_out] if state.k_counts[k] >= 7)
                    t_in_tight = sum(1 for k in triple_to_k222s[t_in] if state.k_counts[k] >= 7)
                    
                    gain = t_out_tight - t_in_tight
                    if gain > best_gain:
                        best_gain = gain
                        best_t_in = t_in
                
                if best_t_in is not None and best_gain > 0:
                    state.remove(t_out)
                    if state.add(best_t_in):
                        improved = True
                        swap_found = True
                        # Update blocking scores
                        present = state.present_list()
                        for t_id in present:
                            blocking_score[t_id] = sum(1 for k in triple_to_k222s[t_id] if state.k_counts[k] >= 7)
                        break
                    else:
                        state.add(t_out)
            
            if not swap_found:
                break
        
        return improved
    
    best_size = 0
    best_present = None
    
    # Greedy initialization by K(2,2,2) count (least constrained first)
    def greedy_init(seed):
        local_rng = random.Random(seed)
        state = State()
        order = list(range(num_triples))
        local_rng.shuffle(order)
        # Sort by number of K(2,2,2)s (fewer constraints first)
        order.sort(key=lambda t: triple_k222_count[t])
        # Add some randomness
        for i in range(len(order)):
            if i > 0 and i < len(order):
                swap_idx = i - local_rng.randint(0, min(i, 20))
                order[i], order[swap_idx] = order[swap_idx], order[i]
        for t_id in order:
            if time.time() > DEADLINE:
                break
            state.add(t_id)
        polish(state)
        return state
    
    # Main loop: multiple restarts
    restart_count = 0
    while time.time() < DEADLINE:
        restart_count += 1
        seed = 42 + restart_count * 7919
        
        if restart_count <= 8:
            # Greedy init
            state = greedy_init(seed)
        else:
            # Perturb best solution
            state = State()
            for i in range(num_triples):
                if best_present[i]:
                    state.present[i] = True
                    state.size += 1
            for k_idx in range(num_k222s):
                for t_id in k222_triples[k_idx]:
                    if state.present[t_id]:
                        state.k_counts[k_idx] += 1
            # Perturb: remove and re-add some
            n_perturb = min(30, state.size // 8)
            present = state.present_list()
            for _ in range(n_perturb):
                if present:
                    t = random.choice(present)
                    state.remove(t)
                    present = state.present_list()
            polish(state)
        
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
        
        # Local search: mix of random swaps and SAT-based search
        n_swaps = 300
        for _ in range(n_swaps):
            if time.time() > DEADLINE:
                break
            r = random.random()
            if r < 0.3:
                success = random_1swap(state)
            elif r < 0.6:
                success = random_2swap(state)
            elif r < 0.8:
                success = random_3swap(state)
            else:
                success = sat_based_local_search(state, max_iters=100)
            
            if success:
                polish(state)
                if state.size > best_size:
                    best_size = state.size
                    best_present = state.present[:]
        
        # Final polish
        polish(state)
        polish_full(state)
        if state.size > best_size:
            best_size = state.size
            best_present = state.present[:]
    
    return [list(triples[t_id]) for t_id in range(num_triples) if best_present[t_id]]
