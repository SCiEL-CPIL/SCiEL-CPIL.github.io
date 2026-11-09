import struct

# CMS Parameters
CMS_ROWS = 30
CMS_COLS = 8
CMS_SIZE = CMS_ROWS * CMS_COLS
CMS_BYTES = CMS_SIZE * 4
HH_SIZE = 16
HH_BYTES = HH_SIZE * 4
TOTAL_BYTES = CMS_BYTES + HH_BYTES # 960 + 64 = 1024

# Hash parameters for CMS
# We need CMS_ROWS pairs of (multiplier, offset)
# Using large odd numbers
PRIMES = [
    1035493289, 775374203, 1495862053, 2015145641, 1764051933,
    1855357761, 1873343063, 1908022841, 1941683001, 1975343161,
    2009003321, 2042663481, 2076323641, 2109983801, 2143643961,
    2177304121, 2210964281, 2244624441, 2278284601, 2311944761,
    2345604921, 2379265081, 2412925241, 2446585401, 2480245561,
    2513905721, 2547565881, 2581226041, 2614886201, 2648546361
]
OFFSETS = [31, 63, 97, 131, 165, 199, 233, 267, 301, 335, 369, 403, 437, 471, 505, 539, 573, 607, 641, 675, 709, 743, 777, 811, 845, 879, 913, 947, 981, 1015]

def _cms_query(cms_data, item):
    min_val = float('inf')
    for r in range(CMS_ROWS):
        idx = ((item * PRIMES[r] + OFFSETS[r]) % 2147483647) % CMS_COLS
        val = struct.unpack_from('<I', cms_data, r * CMS_COLS * 4 + idx * 4)[0]
        if val < min_val:
            min_val = val
    return min_val

def _cms_update(cms_data, item):
    for r in range(CMS_ROWS):
        idx = ((item * PRIMES[r] + OFFSETS[r]) % 2147483647) % CMS_COLS
        offset = r * CMS_COLS * 4 + idx * 4
        val = struct.unpack_from('<I', cms_data, offset)[0]
        val += 1
        struct.pack_into('<I', cms_data, offset, val)

def update(state, item):
    # State is bytes, convert to bytearray for mutation
    state = bytearray(state)
    
    # Split state
    cms_data = state[HH_BYTES:] # 960 bytes
    hh_data = state[:HH_BYTES] # 64 bytes
    
    # Update CMS
    _cms_update(cms_data, item)
    
    # Update HH
    # Parse HH entries
    # Each entry: 2 bytes item, 2 bytes count
    hh_entries = []
    min_idx = 0
    min_count = float('inf')
    
    for i in range(HH_SIZE):
        offset = i * 4
        item_id = struct.unpack_from('<H', hh_data, offset)[0]
        count = struct.unpack_from('<H', hh_data, offset + 2)[0]
        hh_entries.append((item_id, count))
        if count < min_count:
            min_count = count
            min_idx = i
            
    # Check if item is in HH
    found = False
    for i, (item_id, count) in enumerate(hh_entries):
        if item_id == item:
            # Increment count
            new_count = count + 1
            # Check for overflow (shouldn't happen, max 1024)
            struct.pack_into('<H', hh_data, i * 4 + 2, new_count)
            found = True
            break
            
    if not found:
        # Get CMS estimate for item
        est = _cms_query(cms_data, item)
        # est is the count including the current item
        if est > min_count:
            # Replace min entry
            offset = min_idx * 4
            struct.pack_into('<H', hh_data, offset, item)
            struct.pack_into('<H', hh_data, offset + 2, est)
            
    return bytes(state)

def query(state, item):
    # State is bytes
    hh_data = state[:HH_BYTES]
    cms_data = state[HH_BYTES:]
    
    # Check HH
    for i in range(HH_SIZE):
        offset = i * 4
        item_id = struct.unpack_from('<H', hh_data, offset)[0]
        if item_id == item:
            count = struct.unpack_from('<H', hh_data, offset + 2)[0]
            return count
            
    # Fallback to CMS
    return _cms_query(cms_data, item)
