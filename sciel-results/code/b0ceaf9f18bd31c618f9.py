_ALL_MASKS_SORTED = None
def get_sorted_masks():
    global _ALL_MASKS_SORTED
    if _ALL_MASKS_SORTED is None:
        _ALL_MASKS_SORTED = list(range(1 << 16))
        _ALL_MASKS_SORTED.sort(key=lambda x: x.bit_count(), reverse=True)
    return _ALL_MASKS_SORTED
