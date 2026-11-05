    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            # ... add comparator
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    