<<<<<<< SEARCH
    # Pass 3: Local Search on the best chain found
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=12000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Deterministic polish
        if _time_left() > 25:
            polished = _polish(sequence, best_chain, max_iterations=6000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 5: Additional local search with different seed if time permits
        if _time_left() > 35:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=6000,
                                       time_limit=local_search_time * 0.7, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 25:
                final_polish = _polish(sequence, best_chain, max_iterations=4000,
                                       time_limit=polish_time * 0.6, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish
            
            # Pass 5b: High-precision polish with tighter tolerance
            if _time_left() > 20:
                high_prec_polish = _polish(sequence, best_chain, max_iterations=8000,
                                           time_limit=polish_time * 0.8, rng=rng2)
                if high_prec_polish is not None:
                    hp_energy = _energy(sequence, high_prec_polish)
                    if hp_energy < best_energy:
                        best_energy = hp_energy
                        best_chain = high_prec_polish

        # Pass 6: Another round of growth + search if time permits
        if _time_left() > 40:
            extra_grow = 40
            for i in range(extra_grow):
                if _time_left() < 30:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick polish
                    if _time_left() > 25:
                        polished = _polish(sequence, best_chain, max_iterations=2500,
                                           time_limit=1.5, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished
=======
    # Pass 3: Local Search on the best chain found (now faster with incremental energy)
    if best_chain is not None and _time_left() > 30:
        improved = _local_search(sequence, best_chain, max_iterations=25000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Deterministic polish (now faster)
        if _time_left() > 25:
            polished = _polish(sequence, best_chain, max_iterations=12000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 5: Additional local search with different seed if time permits
        if _time_left() > 35:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=15000,
                                       time_limit=local_search_time * 0.8, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 25:
                final_polish = _polish(sequence, best_chain, max_iterations=8000,
                                       time_limit=polish_time * 0.7, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish
            
            # Pass 5b: High-precision polish with tighter tolerance
            if _time_left() > 20:
                high_prec_polish = _polish(sequence, best_chain, max_iterations=10000,
                                           time_limit=polish_time * 0.9, rng=rng2)
                if high_prec_polish is not None:
                    hp_energy = _energy(sequence, high_prec_polish)
                    if hp_energy < best_energy:
                        best_energy = hp_energy
                        best_chain = high_prec_polish

        # Pass 6: Another round of growth + search if time permits
        if _time_left() > 40:
            extra_grow = 60
            for i in range(extra_grow):
                if _time_left() < 30:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick polish
                    if _time_left() > 25:
                        polished = _polish(sequence, best_chain, max_iterations=4000,
                                           time_limit=2.0, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished
>>>>>>> REPLACE
