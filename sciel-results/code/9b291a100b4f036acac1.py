import numpy as np
import time, random, itertools, sys

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_masks = np.array([(1<<a)|(1<<b)|(1<<c) for a,b,c in all_triples], dtype=np.int64)
    
    # Precompute pair indices
    pair_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_indices[(i,j)] = idx
            idx += 1
    num_pairs = n*(n-1)//2  # 120
    
    # For each triple, store which pairs it uses as a frozenset of pair indices
    triple_pairs = []
    for a,b,c in all_triples:
        ps = frozenset([pair_indices[(min(a,b),max(a,b))], 
                        pair_indices[(min(a,c),max(a,c))], 
                        pair_indices[(min(b,c),max(b,c))]])
        triple_pairs.append(ps)
    
    # Precompute popcount
    pop = np.zeros(1<<n, dtype=np.int8)
    for i in range(1, 1<<n):
        pop[i] = pop[i>>1] + (i&1)
    
    def compute_independence(triple_mask_list):
        """Compute independence number given list of triple masks (int64)."""
        if not triple_mask_list:
            return n
        covered = np.zeros(1<<n, dtype=bool)
        for tm in triple_mask_list:
            tm = int(tm)
            comp = ((1<<n) - 1) ^ tm
            sub = comp
            while True:
                covered[tm | sub] = True
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        uncovered = ~covered
        return int(pop[uncovered].max())
    
    def make_indep_counter():
        return np.zeros(1<<n, dtype=np.int16)
    
    def add_triple_to_counter(count, tm):
        tm = int(tm)
        comp = ((1<<n) - 1) ^ tm
        sub = comp
        while True:
            count[tm | sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
    
    def remove_triple_from_counter(count, tm):
        tm = int(tm)
        comp = ((1<<n) - 1) ^ tm
        sub = comp
        while True:
            count[tm | sub] -= 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
    
    def get_indep_from_counter(count):
        zero_mask = (count == 0)
        return int(pop[zero_mask].max())
    
    def greedy_packing(order):
        sys_set = set()
        used_pairs = set()
        for idx in order:
            ps = triple_pairs[idx]
            if all(p not in used_pairs for p in ps):
                sys_set.add(idx)
                used_pairs.update(ps)
        return sys_set
    
    random.seed(42)
    
    best_sys = None
    best_indep = n + 1
    
    # Phase 1: Multiple greedy starts
    for attempt in range(300):
        if time.time() - t0 > DEADLINE * 0.1:
            break
        order = list(range(num_triples))
        random.shuffle(order)
        sys_set = greedy_packing(order)
        tm_list = [triple_masks[i] for i in sys_set]
        indep = compute_independence(tm_list)
        if indep < best_indep:
            best_indep = indep
            best_sys = list(sys_set)
            if best_indep <= 6:
                break
    
    # Phase 2: Simulated annealing
    if best_sys is not None:
        sys_set = set(best_sys)
        cur_indep = best_indep
        
        count = make_indep_counter()
        for i in sys_set:
            add_triple_to_counter(count, triple_masks[i])
        
        T_start = 2.0
        T_end = 0.01
        total_sa_steps = 500000
        
        step = 0
        no_improve = 0
        max_no_improve = 8000
        
        while time.time() - t0 < DEADLINE:
            step += 1
            T = T_start * (T_end / T_start) ** (step / total_sa_steps)
            if T < T_end:
                T = T_end
            
            r = random.random()
            
            if r < 0.3:
                # Add a random valid triple
                cand = random.randrange(num_triples)
                if cand in sys_set:
                    continue
                ps = triple_pairs[cand]
                conflict = False
                for ti in sys_set:
                    if triple_pairs[ti] & ps:
                        conflict = True
                        break
                if conflict:
                    continue
                
                sys_set.add(cand)
                add_triple_to_counter(count, triple_masks[cand])
                new_indep = get_indep_from_counter(count)
                delta = new_indep - cur_indep
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    cur_indep = new_indep
                    no_improve = 0
                    if cur_indep < best_indep:
                        best_indep = cur_indep
                        best_sys = list(sys_set)
                else:
                    sys_set.discard(cand)
                    remove_triple_from_counter(count, triple_masks[cand])
                    no_improve += 1
                    
            elif r < 0.55:
                # Remove a random triple
                if not sys_set:
                    continue
                rem = random.choice(list(sys_set))
                sys_set.discard(rem)
                remove_triple_from_counter(count, triple_masks[rem])
                new_indep = get_indep_from_counter(count)
                delta = new_indep - cur_indep
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    cur_indep = new_indep
                    no_improve = 0
                    if cur_indep < best_indep:
                        best_indep = cur_indep
                        best_sys = list(sys_set)
                else:
                    sys_set.add(rem)
                    add_triple_to_counter(count, triple_masks[rem])
                    no_improve += 1
                    
            elif r < 0.85:
                # Swap: remove one, add another
                if not sys_set:
                    continue
                rem = random.choice(list(sys_set))
                cand = random.randrange(num_triples)
                if cand == rem or cand in sys_set:
                    continue
                
                sys_set.discard(rem)
                remove_triple_from_counter(count, triple_masks[rem])
                
                ps_cand = triple_pairs[cand]
                conflict = False
                for ti in sys_set:
                    if triple_pairs[ti] & ps_cand:
                        conflict = True
                        break
                
                if not conflict:
                    sys_set.add(cand)
                    add_triple_to_counter(count, triple_masks[cand])
                    new_indep = get_indep_from_counter(count)
                    delta = new_indep - cur_indep
                    
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        cur_indep = new_indep
                        no_improve = 0
                        if cur_indep < best_indep:
                            best_indep = cur_indep
                            best_sys = list(sys_set)
                    else:
                        sys_set.discard(cand)
                        remove_triple_from_counter(count, triple_masks[cand])
                        sys_set.add(rem)
                        add_triple_to_counter(count, triple_masks[rem])
                        no_improve += 1
                else:
                    sys_set.add(rem)
                    add_triple_to_counter(count, triple_masks[rem])
                    no_improve += 1
                    
            else:
                # 2-swap: remove two, add two
                if len(sys_set) < 2:
                    continue
                rem1, rem2 = random.sample(list(sys_set), 2)
                cand1 = random.randrange(num_triples)
                cand2 = random.randrange(num_triples)
                
                if cand1 in sys_set or cand2 in sys_set or cand1 == cand2:
                    continue
                
                sys_set.discard(rem1)
                sys_set.discard(rem2)
                remove_triple_from_counter(count, triple_masks[rem1])
                remove_triple_from_counter(count, triple_masks[rem2])
                
                ps1 = triple_pairs[cand1]
                ps2 = triple_pairs[cand2]
                if ps1 & ps2:
                    sys_set.add(rem1)
                    sys_set.add(rem2)
                    add_triple_to_counter(count, triple_masks[rem1])
                    add_triple_to_counter(count, triple_masks[rem2])
                    no_improve += 1
                    continue
                
                conflict = False
                for ti in sys_set:
                    if (triple_pairs[ti] & ps1) or (triple_pairs[ti] & ps2):
                        conflict = True
                        break
                
                if not conflict:
                    sys_set.add(cand1)
                    sys_set.add(cand2)
                    add_triple_to_counter(count, triple_masks[cand1])
                    add_triple_to_counter(count, triple_masks[cand2])
                    new_indep = get_indep_from_counter(count)
                    delta = new_indep - cur_indep
                    
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        cur_indep = new_indep
                        no_improve = 0
                        if cur_indep < best_indep:
                            best_indep = cur_indep
                            best_sys = list(sys_set)
                    else:
                        sys_set.discard(cand1)
                        sys_set.discard(cand2)
                        remove_triple_from_counter(count, triple_masks[cand1])
                        remove_triple_from_counter(count, triple_masks[cand2])
                        sys_set.add(rem1)
                        sys_set.add(rem2)
                        add_triple_to_counter(count, triple_masks[rem1])
                        add_triple_to_counter(count, triple_masks[rem2])
                        no_improve += 1
                else:
                    sys_set.add(rem1)
                    sys_set.add(rem2)
                    add_triple_to_counter(count, triple_masks[rem1])
                    add_triple_to_counter(count, triple_masks[rem2])
                    no_improve += 1
            
            if no_improve > max_no_improve:
                no_improve = 0
                sys_set = set(best_sys)
                count = make_indep_counter()
                for i in sys_set:
                    add_triple_to_counter(count, triple_masks[i])
                cur_indep = best_indep
        
        # Final: greedily add more triples to best solution
        if best_sys is not None:
            sys_set = set(best_sys)
            used_pairs = set()
            for i in sys_set:
                used_pairs.update(triple_pairs[i])
            
            order = list(range(num_triples))
            random.shuffle(order)
            for idx in order:
                if idx in sys_set:
                    continue
                ps = triple_pairs[idx]
                if all(p not in used_pairs for p in ps):
                    sys_set.add(idx)
                    used_pairs.update(ps)
            
            tm_list = [triple_masks[i] for i in sys_set]
            indep = compute_independence(tm_list)
            if indep < best_indep:
                best_indep = indep
                best_sys = list(sys_set)
    
    out = [list(all_triples[i]) for i in sorted(best_sys)]
    return out
