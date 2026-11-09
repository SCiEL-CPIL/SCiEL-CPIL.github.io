            paths = get_paths(i, j)
            # Sort by number of candidate edges
            paths.sort(key=lambda p: sum(1 for e in p if e != -1))
            # Keep only top 5
            paths = paths[:5]
