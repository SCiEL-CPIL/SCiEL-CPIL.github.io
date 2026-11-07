import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Total prize per item
    total_prizes = prizes.sum(axis=0)
    
    # Calculate the "competition" or "prize mass" for each constraint
    # L_c = sum of (total_prize_j * weight_jc) for all j
    # This represents the total prize "competing" for the capacity of constraint c.
    # Shape of total_prizes is (100,)
    # Shape of weights[c] is (100,)
    
    cost = np.zeros(100)
    
    for c in range(5):
        # Total prize-weight for constraint c
        L_c = np.dot(total_prizes, weights[c, :])
        
        # The prompt says "sum of prizes of all OTHER items".
        # So for item i, the competition is L_c - (prize_i * weight_ic)
        # Let's compute the vector of competition for all items in constraint c
        # comp_i = L_c - total_prizes[i] * weights[c, i]
        # We can vectorize this:
        comp = L_c - total_prizes * weights[c, :]
        
        # The cost to item i is its weight in c times the competition from others
        # cost_i += weights[c, i] * comp_i
        cost += weights[c, :] * comp
    
    epsilon = 1e-9
    heur = total_prizes / (cost + epsilon)
    
    return heur
