from itertools import combinations
import random

def construct():
    all_sets = []
    for comb in combinations(range(16), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_sets.append(mask)
    
    best_chosen = []
    
    for seed in range(20):
        random.seed(seed)
        shuffled = all_sets[:]
        random.shuffle(shuffled)
        
        chosen = []
        inter_dicts = []
        
        for mask in shuffled:
            conflict = False
            n = len(chosen)
            for i in range(n):
                iac = chosen[i] & mask
                if iac in inter_dicts[i]:
                    for j in inter_dicts[i][iac]:
                        if (chosen[j] & mask) == iac:
                            conflict = True
                            break
                if conflict:
                    break
            if not conflict:
                idx = len(chosen)
                chosen.append(mask)
                new_dict = {}
                for i in range(idx):
                    inter_val = chosen[i] & mask
                    if inter_val not in inter_dicts[i]:
                        inter_dicts[i][inter_val] = []
                    inter_dicts[i][inter_val].append(idx)
                    if inter_val not in new_dict:
                        new_dict[inter_val] = []
                    new_dict[inter_val].append(i)
                inter_dicts.append(new_dict)
        
        if len(chosen) > len(best_chosen):
            best_chosen = chosen
    
    result = []
    for mask in best_chosen:
        s = sorted(i+1 for i in range(16) if mask & (1 << i))
        result.append(s)
    return result
