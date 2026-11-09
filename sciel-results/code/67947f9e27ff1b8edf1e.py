<<<<<<< SEARCH
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = _local_search(adjacency, init_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 3: Simulated annealing from the best local search result
    if time.time() < deadline - 10.0:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Phase 4: Tabu search from the best result
    if time.time() < deadline - 5.0:
        tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, deadline)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels
    
    # Final check
    if time.time() < deadline - 2.0:
        final_labels = _local_search(adjacency, best_labels, deadline)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    return best_labels
=======
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = _local_search(adjacency, init_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 3: Simulated annealing from the best local search result
    if time.time() < deadline - 15.0:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline - 10.0)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        # Final local search after SA
        polished = _local_search(adjacency, best_labels, deadline - 5.0)
        polished_cut = _cut_size(adjacency, polished)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished
    
    # Phase 4: Tabu search from the best result
    if time.time() < deadline - 10.0:
        tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, deadline - 5.0)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels
        # Final local search after tabu
        polished = _local_search(adjacency, best_labels, deadline - 2.0)
        polished_cut = _cut_size(adjacency, polished)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished
    
    # Final high-precision polish: run local search with tighter tolerance
    if time.time() < deadline - 1.0:
        final_labels = _local_search(adjacency, best_labels, deadline - 0.5)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    return best_labels
>>>>>>> REPLACE
