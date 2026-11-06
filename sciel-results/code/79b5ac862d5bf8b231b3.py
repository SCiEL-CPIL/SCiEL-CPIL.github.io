<<<<<<< SEARCH
    # Phase 2: Splicing (20% of time)
    deadline2 = start_time + time_left() * 0.9
    if len(all_paths) >= 2:
        # Try splicing pairs
        tried = set()
        while time.time() < deadline2:
            i = random.randint(0, len(all_paths) - 1)
            j = random.randint(0, len(all_paths) - 1)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in tried:
                continue
            tried.add(key)
            
            spliced = try_splice(all_paths[i], all_paths[j])
            if spliced and len(spliced) > best_len:
                # Try to extend the spliced path greedily
                spliced_ext = greedy_extend(spliced, 'forward', max_steps=20)
                if len(spliced_ext) > len(spliced):
                    spliced = spliced_ext
                spliced_ext = greedy_extend(spliced, 'backward', max_steps=20)
                if len(spliced_ext) > len(spliced):
                    spliced = spliced_ext
                
                if len(spliced) > best_len:
                    best_len = len(spliced)
                    best_path = spliced
                    all_paths.append(spliced)
                    if len(all_paths) > 20:
                        all_paths.sort(key=lambda x: -len(x))
                        all_paths = all_paths[:20]
=======
    # Phase 2: Splicing (20% of time)
    deadline2 = start_time + time_left() * 0.9
    if len(all_paths) >= 2:
        # Try splicing pairs systematically
        tried = set()
        n_paths = len(all_paths)
        while time.time() < deadline2:
            i = random.randint(0, n_paths - 1)
            j = random.randint(0, n_paths - 1)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in tried:
                continue
            tried.add(key)
            
            spliced = try_splice(all_paths[i], all_paths[j])
            if spliced and len(spliced) > best_len:
                # Try to extend the spliced path greedily from both ends
                spliced_ext = greedy_extend(spliced, 'forward', max_steps=30)
                if len(spliced_ext) > len(spliced):
                    spliced = spliced_ext
                spliced_ext = greedy_extend(spliced, 'backward', max_steps=30)
                if len(spliced_ext) > len(spliced):
                    spliced = spliced_ext
                
                if len(spliced) > best_len:
                    best_len = len(spliced)
                    best_path = spliced
                    all_paths.append(spliced)
                    n_paths = len(all_paths)
                    if len(all_paths) > 25:
                        all_paths.sort(key=lambda x: -len(x))
                        all_paths = all_paths[:25]
                        n_paths = len(all_paths)
>>>>>>> REPLACE

