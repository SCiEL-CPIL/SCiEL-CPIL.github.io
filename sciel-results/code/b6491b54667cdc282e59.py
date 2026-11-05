    def reduce_blocks_fast(indices):
        sel_set = set(indices)
        
        # Calculate cover counts
        cover_count = {}
        for bi in sel_set:
            for ts in block_tsubs[bi]:
                cover_count[ts] = cover_count.get(ts, 0) + 1
                
        changed = True
        while changed:
            changed = False
            items = list(sel_set)
            random.shuffle(items)
            for bi in items:
                if bi not in sel_set:
                    continue
                # Check if bi is redundant
                redundant = True
                for ts in block_tsubs[bi]:
                    if cover_count[ts] <= 1:
                        redundant = False
                        break
                if redundant:
                    sel_set.remove(bi)
                    for ts in block_tsubs[bi]:
                        cover_count[ts] -= 1
                    changed = True
        return list(sel_set)
