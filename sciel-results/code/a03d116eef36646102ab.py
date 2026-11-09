"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 16
    all_triples = list(combinations(range(n), 3))
    T = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    triple_set = set(all_triples)
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is 3 disjoint pairs giving 8 transversal triples
    # For triple (a,b,c), partners x'!=a, y'!=b, z'!=c
    # 8 transversals must all be valid triples
    
    k222_configs = []
    k222_seen = set()
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                # triple (a,b,c)
                for xp in range(n):
                    if xp == a: continue
                    for yp in range(n):
                        if yp == b or yp == a: continue
                        for zp in range(n):
                            if zp == c or zp == a or zp == b: continue
                            # Check 8 transversals
                            trans = []
                            ok = True
                            for i in range(2):
                                for j in range(2):
                                    for k in range(2):
                                        p1 = a if i==0 else xp
                                        p2 = b if j==0 else yp
                                        p3 = c if k==0 else zp
                                        s = set((p1,p2,p3))
                                        if len(s) != 3:
                                            ok = False
                                            break
                                        t = tuple(sorted((p1,p2,p3)))
                                        if t not in triple_set:
                                            ok = False
                                            break
                                        trans.append(t)
                                    if not ok: break
                                if not ok: break
                            if not ok: continue
                            config = tuple(sorted(triple_to_idx[t] for t in trans))
                            if config not in k222_seen:
                                k222_seen.add(config)
                                k222_configs.append(config)
    
    K = len(k222_configs)
    
    # Bitmasks for each K(2,2,2)
    k222_masks = []
    for config in k222_configs:
        mask = 0
        for idx in config:
            mask |= (1 << idx)
        k222_masks.append(mask)
    
    # Count K(2,2,2) per triple
    t_k_count = [0] * T
    for config in k222_configs:
        for idx in config:
            t_k_count[idx] += 1
    
    def eval_chrom(chrom):
        """Return (count, violations)"""
        cnt = chrom.bit_count()
        viols = 0
        for mask in k222_masks:
            if (chrom & mask) == mask:
                viols += 1
        return cnt, viols
    
    def fit(chrom):
        cnt, viols = eval_chrom(chrom)
        return cnt - 1000 * viols
    
    # Initialize
    pop_size = 300
    pop = []
    for _ in range(pop_size):
        chrom = 0
        for i in range(T):
            p = 0.2 + 0.3 * (t_k_count[i] / max(1, max(t_k_count)))
            if random.random() < p:
                chrom |= (1 << i)
        pop.append(chrom)
    
    best_chrom = max(pop, key=fit)
    best_fit = fit(best_chrom)
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        scored = sorted([(fit(c), c) for c in pop], reverse=True)
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best_chrom = scored[0][1]
        
        new_pop = [scored[i][1] for i in range(max(5, pop_size//10))]
        
        def tour():
            return max(random.sample(scored, 3), key=lambda x: x[0])[1]
        
        while len(new_pop) < pop_size:
            p1, p2 = tour(), tour()
            child = 0
            for i in range(T):
                if random.random() < 0.5:
                    if (p1 >> i) & 1:
                        child |= (1 << i)
                else:
                    if (p2 >> i) & 1:
                        child |= (1 << i)
            # Mutation
            for i in range(T):
                if random.random() < 0.03:
                    child ^= (1 << i)
            new_pop.append(child)
        
        pop = new_pop
    
    result = [list(all_triples[i]) for i in range(T) if (best_chrom >> i) & 1]
    
    # Verify validity; if invalid, greedy repair
    def check_valid(triples):
        tset = set(tuple(t) for t in triples)
        present = set()
        for t in triples:
            tt = tuple(t)
            if tt in triple_set:
                present.add(triple_to_idx[tt])
        for config in k222_configs:
            if all(idx in present for idx in config):
                return False
        return True
    
    if not check_valid(result):
        # Greedy repair: remove triples causing violations
        tset = set(tuple(t) for t in result)
        present = set()
        for t in result:
            tt = tuple(t)
            if tt in triple_set:
                present.add(triple_to_idx[tt])
        
        changed = True
        while changed:
            changed = False
            for config in k222_configs:
                if all(idx in present for idx in config):
                    # Remove the triple with highest k222_count
                    worst = max(config, key=lambda i: t_k_count[i])
                    present.discard(worst)
                    tset.discard(all_triples[worst])
                    changed = True
                    break
        
        result = [list(t) for t in tset]
    
    return result
# EVOLVE-BLOCK-END
