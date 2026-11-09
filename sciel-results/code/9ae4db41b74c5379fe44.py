    def local_search_add():
        order = list(range(n_total))
        random.shuffle(order)
        changed = True
        while changed:
            changed = False
            for idx in order:
                if not in_family[idx] and block_count[idx] == 0:
                    add_set(idx)
                    changed = True
