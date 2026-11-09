<<<<<<< SEARCH
    def local_search_swap(selected_mask, rng, max_iters=50):
        """
        Local search: try to swap out an element for another that increases size.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        current_mask = selected_mask

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Get list of selected elements
            selected_list = []
            temp = current_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                selected_list.append(v)
                temp ^= lb

            if not selected_list:
                break

            # Pick a random element to remove
            remove_elem = rng.choice(selected_list)
            current_mask &= ~(1 << remove_elem)

            # Try to add a random element not in current_mask
            candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
            if not candidates:
                current_mask |= (1 << remove_elem)
                continue

            rng.shuffle(candidates)
            added = False
            for add_elem in candidates[:40]:
                # Check if add_elem is compatible with all in current_mask
                compatible = True
                temp = current_mask
                while temp:
                    lb = temp & (-temp)
                    v = lb.bit_length() - 1
                    if (conflict_mask[add_elem] >> v) & 1:
                        compatible = False
                        break
                    temp ^= lb
                if compatible:
                    current_mask |= (1 << add_elem)
                    added = True
                    if bin(current_mask).count('1') > best_size:
                        best_size = bin(current_mask).count('1')
                        best_mask = current_mask
                    break

            if not added:
                current_mask |= (1 << remove_elem)

        return best_mask
=======
    def local_search_swap(selected_mask, rng, max_iters=50):
        """
        Local search with neutral-swap acceptance: try to swap out an element for another.
        Accept swaps that increase size, and also accept neutral swaps (same size) with
        some probability to explore the search space. After each swap, try to add elements.
        """
        best_mask = selected_mask
        best_size = bin(selected_mask).count('1')
        current_mask = selected_mask

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Get list of selected elements
            selected_list = []
            temp = current_mask
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                selected_list.append(v)
                temp ^= lb

            if not selected_list:
                break

            # Try multiple swap attempts in this iteration
            for _ in range(5):
                if time.time() > deadline:
                    break

                # Pick a random element to remove
                remove_elem = rng.choice(selected_list)
                current_mask &= ~(1 << remove_elem)

                # Try to add a random element not in current_mask
                candidates = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                if not candidates:
                    current_mask |= (1 << remove_elem)
                    continue

                rng.shuffle(candidates)
                added = False
                for add_elem in candidates[:50]:
                    # Check if add_elem is compatible with all in current_mask
                    compatible = True
                    temp = current_mask
                    while temp:
                        lb = temp & (-temp)
                        v = lb.bit_length() - 1
                        if (conflict_mask[add_elem] >> v) & 1:
                            compatible = False
                            break
                        temp ^= lb
                    if compatible:
                        current_mask |= (1 << add_elem)
                        added = True
                        cur_size = bin(current_mask).count('1')
                        if cur_size > best_size:
                            best_size = cur_size
                            best_mask = current_mask
                        # Accept neutral swaps with some probability
                        elif cur_size == best_size and rng.random() < 0.3:
                            best_mask = current_mask
                        break

                if not added:
                    current_mask |= (1 << remove_elem)

                # After a successful swap, try to add more elements
                if added:
                    candidates2 = [x for x in range(1, N + 1) if not (current_mask >> x) & 1]
                    rng.shuffle(candidates2)
                    for add_elem in candidates2[:30]:
                        compatible = True
                        temp = current_mask
                        while temp:
                            lb = temp & (-temp)
                            v = lb.bit_length() - 1
                            if (conflict_mask[add_elem] >> v) & 1:
                                compatible = False
                                break
                            temp ^= lb
                        if compatible:
                            current_mask |= (1 << add_elem)
                            cur_size = bin(current_mask).count('1')
                            if cur_size > best_size:
                                best_size = cur_size
                                best_mask = current_mask
                        else:
                            continue
                    # Update selected_list after additions
                    selected_list = []
                    temp = current_mask
                    while temp:
                        lb = temp & (-temp)
                        v = lb.bit_length() - 1
                        selected_list.append(v)
                        temp ^= lb

        return best_mask
>>>>>>> REPLACE

