<<<<<<< SEARCH
def _greedy_polish(sequence, chain, max_rounds=100):
    """
    Greedy polish: repeatedly scan all possible pivot and crankshaft moves and accept any that improve energy.
    Stop when no move improves energy or max_rounds reached.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        
        # Try pivot moves
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        # Try crankshaft moves
        if n > 6:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy
=======
def _greedy_polish(sequence, chain, max_rounds=100):
    """
    Greedy polish: repeatedly scan all possible pivot, crankshaft, and two-point swap moves 
    and accept any that improve energy. Stop when no move improves energy or max_rounds reached.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    for _ in range(max_rounds):
        improved = False
        
        # Try pivot moves
        for k in range(n - 1):
            for rot_choice in range(3):
                new_chain, new_energy = _try_pivot_move(sequence, current_chain, k, rot_choice)
                if new_chain is not None and new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    improved = True
                    break
            if improved:
                break
        
        # Try crankshaft moves
        if n > 6 and not improved:
            for k1 in range(n - 2):
                for k2 in range(k1 + 2, n):
                    for rot_choice in range(3):
                        new_chain, new_energy = _try_crankshaft(sequence, current_chain, k1, k2, rot_choice)
                        if new_chain is not None and new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Try two-point swaps
        if not improved:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_chain, new_energy = _try_two_point_swap(sequence, current_chain, i, j)
                    if new_chain is not None and new_energy < current_energy:
                        current_chain = new_chain
                        current_energy = new_energy
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
            
    return current_chain, current_energy
>>>>>>> REPLACE

